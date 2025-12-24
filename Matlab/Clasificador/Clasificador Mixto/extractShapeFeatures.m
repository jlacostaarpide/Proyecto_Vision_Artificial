function [feat, featNames, dbg] = extractShapeFeatures(I, opts)
% EXTRACTSHAPEFEATURES  Features "estructurales" (sin Fourier) para LEGO segmentado (fondo negro).
% Diseñadas para generalizar mejor entre vistas y, especialmente, separar clases parecidas (p.ej. 09 vs 12).
%
% Entrada:
%   I    : imagen RGB o gray, pieza ya segmentada con fondo negro.
%   opts : (opcional) struct con parámetros (ver defaults abajo)
%
% Salida:
%   feat      : 1xN double
%   featNames : 1xN cell
%   dbg       : struct con máscaras/diagnóstico (útil para depurar)
%
% Features (N=24):
%   1  AreaNorm
%   2  PerimNorm
%   3  Circularity
%   4  Extent
%   5  Solidity
%   6  Eccentricity
%   7  AspectRatio
%   8  EulerNumber
%   9  HolesCount
%  10  HolesAreaFrac
%  11  SkelLenNorm
%  12  SkelEndpoints
%  13  SkelBranchpoints
%  14  ProjV_peaks
%  15  ProjH_peaks
%  16  ProjV_entropy
%  17  ProjH_entropy
%  18  GridOccFrac_3x3
%  19  GridOccGini_3x3
%  20  GridOccDiagDiff_3x3
%  21  StudsCount
%  22  StudsCountNormArea
%  23  StudsMeanRadius
%  24  StudsRadiusStd
%
% NOTA:
% - Esta función asume que el fondo es negro (0) y la pieza tiene pixeles > 0.
% - Para comparar 9 vs 12, suelen ayudar MUCHO: GridOcc*, Projections*, Holes*, Studs*.

% ---------------- defaults ----------------
if nargin < 2, opts = struct(); end
d.tBlackMin        = 0.03;   % umbral "no negro"
d.minObjArea       = 300;
d.closeRadius      = 3;
d.openRadius       = 2;
d.holeSmallMaxArea = 200;    % solo rellenar agujeros pequeños (ruido)
d.cropTight        = true;

% studs (círculos) – ajusta si hace falta
d.studs_enable     = true;
d.studs_sensitivity= 0.92;   % 0..1 (más alto = detecta más, puede meter falsos)
d.studs_edgeThresh = 0.08;   % para imfindcircles
d.studs_rmin       = 6;      % rango de radios (en px) después de normalizar escala
d.studs_rmax       = 20;

% normalización de escala (recomendado)
d.normTargetSize   = 220;    % tamaño del lado largo del crop (px)

% proyecciones
d.proj_smooth      = 7;      % suavizado movmean

% grid
d.gridN            = 3;      % 3x3

% aplicar overrides
fn = fieldnames(d);
for k=1:numel(fn)
    if isfield(opts, fn{k})
        d.(fn{k}) = opts.(fn{k});
    end
end
opts = d;

% ---------------- 0) preparar ----------------
if ~isfloat(I), I = im2double(I); end
if size(I,3)==3, Ig = rgb2gray(I); else, Ig = I; end

% máscara base "no negro"
mask = Ig > opts.tBlackMin;

% limpieza
mask = bwareaopen(mask, opts.minObjArea);
mask = imclose(mask, strel('disk', opts.closeRadius));
mask = imopen(mask,  strel('disk', opts.openRadius));

% conservar agujeros grandes, rellenar pequeños (ruido)
maskFilled = imfill(mask, 'holes');
holes = maskFilled & ~mask;
holesBig = bwareaopen(holes, opts.holeSmallMaxArea); % deja grandes
holesToFill = holes & ~holesBig;                     % pequeños
mask = mask | holesToFill;

% componente mayor
CC = bwconncomp(mask, 8);
if CC.NumObjects == 0
    feat = zeros(1,24);
    featNames = defaultNames();
    dbg = struct('mask', mask);
    return;
end
if CC.NumObjects > 1
    areas = cellfun(@numel, CC.PixelIdxList);
    [~, imax] = max(areas);
    mask2 = false(size(mask));
    mask2(CC.PixelIdxList{imax}) = true;
    mask = mask2;
end

% orientación + crop
S0 = regionprops(mask, 'Orientation', 'BoundingBox');
ang = -S0.Orientation;
maskR = imrotate(mask, ang, 'nearest', 'loose');
IgR   = imrotate(Ig,   ang, 'bilinear', 'loose');

S1 = regionprops(maskR, 'BoundingBox');
bb = S1.BoundingBox;
if opts.cropTight
    maskR = imcrop(maskR, bb);
    IgR   = imcrop(IgR,   bb);
end

% normalizar escala: llevar a tamaño fijo (lado largo)
h = size(maskR,1); w = size(maskR,2);
scale = opts.normTargetSize / max(h,w);
if ~isfinite(scale) || scale<=0, scale = 1; end
maskN = imresize(maskR, scale, 'nearest');
IgN   = imresize(IgR,   scale, 'bilinear');

% ---------------- 1) props globales ----------------
S = regionprops(maskN, 'Area','Perimeter','Eccentricity','Solidity','Extent', ...
                      'MajorAxisLength','MinorAxisLength','ConvexArea','EulerNumber');
A = S.Area;
P = max(S.Perimeter, 1e-9);

AreaNorm  = A / numel(maskN);                    % [0..1]
PerimNorm = P / max(2*(size(maskN,1)+size(maskN,2)), 1e-9);

Circularity  = (4*pi*A) / (P^2);
Extent       = S.Extent;
Solidity     = S.Solidity;
Eccentricity = S.Eccentricity;

maj = max(S.MajorAxisLength, 1e-9);
minr= max(S.MinorAxisLength, 1e-9);
AspectRatio  = maj / minr;

EulerNumber  = S.EulerNumber;

% agujeros "grandes" (estructura)
maskNFilled = imfill(maskN, 'holes');
holesN = maskNFilled & ~maskN;
CC_h = bwconncomp(holesN, 8);
HolesCount = CC_h.NumObjects;
HolesArea  = nnz(holesN);
HolesAreaFrac = HolesArea / max(A, 1e-9);

% ---------------- 2) skeleton ----------------
skel = bwmorph(maskN, 'skel', Inf);
SkelLenNorm = nnz(skel) / max(sqrt(A), 1e-9);
endp = bwmorph(skel, 'endpoints');
brp  = bwmorph(skel, 'branchpoints');
SkelEndpoints    = nnz(endp);
SkelBranchpoints = nnz(brp);

% ---------------- 3) proyecciones (estructura 1D) ----------------
projV = sum(maskN, 1); % columnas
projH = sum(maskN, 2)';% filas

% suavizar
projV_s = movmean(projV, opts.proj_smooth);
projH_s = movmean(projH, opts.proj_smooth);

% normalizar para entropía
pV = projV_s / max(sum(projV_s), 1e-12);
pH = projH_s / max(sum(projH_s), 1e-12);

ProjV_entropy = -sum(pV .* log(pV + 1e-12));
ProjH_entropy = -sum(pH .* log(pH + 1e-12));

% picos: cuenta máximos locales por encima de umbral relativo
ProjV_peaks = countPeaks(projV_s);
ProjH_peaks = countPeaks(projH_s);

% ---------------- 4) grid occupancy (3x3 por defecto) ----------------
G = gridOccupancy(maskN, opts.gridN); % matriz NxN de fracciones
g = G(:);

GridOccFrac_3x3 = mean(g > 0.15);               % % celdas con “masa”
GridOccGini_3x3 = giniCoeff(g);                  % desigualdad de distribución
GridOccDiagDiff_3x3 = abs(sum(diag(G)) - sum(diag(flipud(G)))); % asimetría diagonal

% ---------------- 5) studs (círculos) ----------------
StudsCount = 0;
StudsCountNormArea = 0;
StudsMeanRadius = 0;
StudsRadiusStd = 0;

if opts.studs_enable
    % Realzar studs: usar gradiente + limitar a máscara
    Ieq = adapthisteq(mat2gray(IgN));
    Gmag = imgradient(Ieq);
    Gmag(~maskN) = 0;

    % imfindcircles necesita imagen 2D
    try
        [centers, radii] = imfindcircles(Gmag, [opts.studs_rmin opts.studs_rmax], ...
            'ObjectPolarity','bright', ...
            'Sensitivity', opts.studs_sensitivity, ...
            'EdgeThreshold', opts.studs_edgeThresh);

        if ~isempty(centers)
            % filtrar círculos cuyo centro esté dentro de la pieza
            cx = round(centers(:,1)); cy = round(centers(:,2));
            ok = cx>=1 & cx<=size(maskN,2) & cy>=1 & cy<=size(maskN,1);
            cx = cx(ok); cy = cy(ok); radii = radii(ok);

            keep = false(numel(radii),1);
            for ii=1:numel(radii)
                keep(ii) = maskN(cy(ii), cx(ii));
            end
            radii = radii(keep);

            % NMS simple: eliminar círculos muy parecidos (reduce duplicados)
            radii = sort(radii);
            radii = unique(round(radii)); %#ok<NASGU>

            StudsCount = numel(keep(keep)); %#ok<*NBRAK> 
            % (alternativa robusta)
            StudsCount = numel(radii);

            StudsCountNormArea = StudsCount / max(A/1e4, 1e-9); % normaliza por área (escala)
            StudsMeanRadius = mean(radii);
            StudsRadiusStd  = std(radii);
        end
    catch
        % si falla imfindcircles por toolbox/parametros, se queda a 0
    end
end

% ---------------- ensamblar ----------------
feat = double([ ...
    AreaNorm, PerimNorm, Circularity, Extent, Solidity, Eccentricity, AspectRatio, EulerNumber, ...
    HolesCount, HolesAreaFrac, SkelLenNorm, SkelEndpoints, SkelBranchpoints, ...
    ProjV_peaks, ProjH_peaks, ProjV_entropy, ProjH_entropy, ...
    GridOccFrac_3x3, GridOccGini_3x3, GridOccDiagDiff_3x3, ...
    StudsCount, StudsCountNormArea, StudsMeanRadius, StudsRadiusStd ...
]);

featNames = defaultNames();

% debug
dbg = struct();
dbg.maskN = maskN;
dbg.IgN   = IgN;
dbg.projV = projV_s;
dbg.projH = projH_s;
dbg.grid  = G;
dbg.skel  = skel;

end

% ================= helpers =================

function names = defaultNames()
names = { ...
 'AreaNorm','PerimNorm','Circularity','Extent','Solidity','Eccentricity','AspectRatio','EulerNumber', ...
 'HolesCount','HolesAreaFrac','SkelLenNorm','SkelEndpoints','SkelBranchpoints', ...
 'ProjV_peaks','ProjH_peaks','ProjV_entropy','ProjH_entropy', ...
 'GridOccFrac_3x3','GridOccGini_3x3','GridOccDiagDiff_3x3', ...
 'StudsCount','StudsCountNormArea','StudsMeanRadius','StudsRadiusStd' ...
};
end

function n = countPeaks(x)
% cuenta máximos locales "relevantes" (umbral relativo)
x = x(:)';
if numel(x) < 5
    n = 0; return;
end
x = x - min(x);
mx = max(x);
if mx < 1e-9
    n = 0; return;
end
x = x / mx;
% máximos locales
isPeak = false(size(x));
for i=2:numel(x)-1
    isPeak(i) = x(i) > x(i-1) && x(i) > x(i+1);
end
% umbral: pico por encima de 0.35 (ajusta si quieres)
n = sum(isPeak & (x > 0.35));
end

function G = gridOccupancy(mask, N)
% devuelve fracción de pixeles "on" por celda en una rejilla NxN
[h,w] = size(mask);
G = zeros(N,N);
ys = round(linspace(1, h+1, N+1));
xs = round(linspace(1, w+1, N+1));
for r=1:N
    for c=1:N
        y1 = ys(r); y2 = ys(r+1)-1;
        x1 = xs(c); x2 = xs(c+1)-1;
        y1 = max(1,y1); x1 = max(1,x1);
        y2 = min(h,y2); x2 = min(w,x2);
        cellMask = mask(y1:y2, x1:x2);
        G(r,c) = nnz(cellMask) / max(numel(cellMask),1);
    end
end
end

function g = giniCoeff(x)
% gini de un vector no-negativo
x = x(:);
x = max(x,0);
s = sum(x);
if s < 1e-12
    g = 0; return;
end
x = sort(x);
n = numel(x);
idx = (1:n)';
g = (2*sum(idx.*x)/(n*s)) - (n+1)/n;
g = abs(g);
end
