function [feat, featNames, dbg] = extractShapeFeatures(I, opts)
% EXTRACTSHAPEFEATURES_6  Versión reducida (6 features) para LEGO segmentado (fondo negro).
%
% Features (N=6):
%   1  AreaNorm
%   2  PerimNorm
%   3  ProjH_entropy
%   4  GridOccFrac_3x3
%   5  GridOccGini_3x3
%   6  GridOccDiagDiff_3x3
%
% Entrada:
%   I    : imagen RGB o gray, pieza segmentada con fondo negro.
%   opts : (opcional) struct con parámetros (defaults abajo).
%
% Salida:
%   feat      : 1x6 double
%   featNames : 1x6 cell
%   dbg       : struct con máscaras/diagnóstico (útil para depurar)

% ---------------- defaults ----------------
if nargin < 2, opts = struct(); end
d.tBlackMin        = 0.03;  % umbral "no negro"
d.minObjArea       = 300;
d.closeRadius      = 3;
d.openRadius       = 2;
d.holeSmallMaxArea = 200;   % rellenar solo agujeros pequeños (ruido)
d.cropTight        = true;

% normalización de escala
d.normTargetSize   = 220;   % tamaño del lado largo del crop (px)

% proyecciones
d.proj_smooth      = 7;     % suavizado movmean

% grid
d.gridN            = 3;     % 3x3

% aplicar overrides
fn = fieldnames(d);
for k=1:numel(fn)
    if isfield(opts, fn{k})
        d.(fn{k}) = opts.(fn{k});
    end
end
opts = d;

featNames = defaultNames6();
dbg = struct();

% ---------------- 0) preparar ----------------
try
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
        feat = zeros(1,6);
        dbg.mask = mask;
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

    % ---------------- 1) props globales (Area/Perimeter) ----------------
    S = regionprops(maskN, 'Area','Perimeter');
    A = S.Area;
    P = max(S.Perimeter, 1e-9);

    AreaNorm  = A / numel(maskN); % [0..1]
    PerimNorm = P / max(2*(size(maskN,1)+size(maskN,2)), 1e-9);

    % ---------------- 2) proyección horizontal: entropía ----------------
    projH = sum(maskN, 2)';           % filas
    projH_s = movmean(projH, opts.proj_smooth);
    pH = projH_s / max(sum(projH_s), 1e-12);
    ProjH_entropy = -sum(pH .* log(pH + 1e-12));

    % ---------------- 3) grid occupancy (3x3) ----------------
    G = gridOccupancy(maskN, opts.gridN); % NxN de fracciones
    g = G(:);

    GridOccFrac_3x3      = mean(g > 0.15);
    GridOccGini_3x3      = giniCoeff(g);
    GridOccDiagDiff_3x3  = abs(sum(diag(G)) - sum(diag(flipud(G))));

    % ---------------- ensamblar ----------------
    feat = double([ ...
        AreaNorm, PerimNorm, ProjH_entropy, ...
        GridOccFrac_3x3, GridOccGini_3x3, GridOccDiagDiff_3x3 ...
    ]);

    % debug
    dbg.maskN = maskN;
    dbg.IgN   = IgN;
    dbg.projH = projH_s;
    dbg.grid  = G;

catch
    feat = zeros(1,6);
    % dbg se queda como esté (vacío o parcial)
end
end

% ================= helpers =================

function names = defaultNames6()
names = { ...
 'AreaNorm','PerimNorm','ProjH_entropy', ...
 'GridOccFrac_3x3','GridOccGini_3x3','GridOccDiagDiff_3x3' ...
};
end

function G = gridOccupancy(mask, N)
% fracción de pixeles "on" por celda en rejilla NxN
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
