function feat = extractShapeFeatures(I)
% EXTRACTSHAPEFEATURES  Features de forma robustas para piezas LEGO segmentadas (fondo negro).
%   Entrada:
%       I : imagen RGB (o gray) de la pieza segmentada con fondo negro.
%   Salida:
%       feat : 1x14 double (vector de features)

    % ------------ Parámetros (ajústalos si lo ves necesario) ------------
    tBlackMin        = 0.03;   % umbral mínimo para considerar "no negro"
    minObjArea       = 300;    % quita motas
    holeSmallMaxArea = 200;    % rellenar solo agujeros pequeños (ruido), conservar agujeros grandes (forma tipo U)
    closeRadius      = 3;      % suaviza bordes mordidos
    openRadius       = 2;      % quita ruido fino
    Nboundary        = 128;    % puntos para contorno (Fourier)
    Kfourier         = 5;      % usaremos magnitudes 2..5 (4 features)
    % -------------------------------------------------------------------

    % 0) Asegurar double
    if ~isfloat(I), I = im2double(I); end
    if size(I,3) == 3
        Ig = rgb2gray(I);
    else
        Ig = I;
    end

    % 1) Máscara robusta: "todo lo que no es negro"
    mask = Ig > tBlackMin;

    % 2) Limpieza morfológica para bordes imperfectos
    mask = bwareaopen(mask, minObjArea);
    mask = imclose(mask, strel('disk', closeRadius));
    mask = imopen(mask,  strel('disk', openRadius));

    % 3) Conservar agujeros grandes (forma), rellenar solo agujeros pequeños (ruido)
    maskFilled = imfill(mask, 'holes');
    holes = maskFilled & ~mask;                 % agujeros detectados
    holesSmall = bwareaopen(holes, holeSmallMaxArea); % elimina agujeros pequeños -> se quedan agujeros grandes
    % holesSmall contiene agujeros "grandes" (porque bwareaopen quita los pequeños)
    % Queremos rellenar SOLO los pequeños => pequeños = holes & ~holesSmall
    holesToFill = holes & ~holesSmall;
    mask = mask | holesToFill;

    % 4) Quedarse con el componente más grande
    CC = bwconncomp(mask, 8);
    if CC.NumObjects == 0
        feat = zeros(1,14);
        return;
    end
    if CC.NumObjects > 1
        areas = cellfun(@numel, CC.PixelIdxList);
        [~, imax] = max(areas);
        mask2 = false(size(mask));
        mask2(CC.PixelIdxList{imax}) = true;
        mask = mask2;
    end

    % 5) Normalizar orientación (reduce errores por rotación)
    S0 = regionprops(mask, 'Orientation', 'BoundingBox');
    ang = -S0.Orientation;
    maskR = imrotate(mask, ang, 'nearest', 'loose');
    % recortar al bounding box del objeto rotado
    S1 = regionprops(maskR, 'BoundingBox');
    bb = S1.BoundingBox;
    maskR = imcrop(maskR, bb);

    % 6) Regionprops robustas (sobre máscara normalizada)
    S = regionprops(maskR, 'Area','Perimeter','Eccentricity','Solidity','Extent', ...
                          'MajorAxisLength','MinorAxisLength','ConvexArea','EulerNumber');

    A  = S.Area;
    P  = max(S.Perimeter, 1e-9);

    circ = (4*pi*A) / (P^2);                       % circularity (robusta si borde no es horroroso)
    ecc  = S.Eccentricity;
    sol  = S.Solidity;
    ext  = S.Extent;

    maj = max(S.MajorAxisLength, 1e-9);
    mino = max(S.MinorAxisLength, 1e-9);
    aspect = maj / mino;

    conv = A / max(S.ConvexArea, 1e-9);            % convexity/compactness
    euler = S.EulerNumber;                         % agujeros grandes (forma tipo U) => muy discriminativo

    % 7) Esqueleto (estructura global)
    skel = bwmorph(maskR, 'skel', Inf);
    skelLen = sum(skel(:));
    skelLenNorm = skelLen / max(sqrt(A), 1e-9);    % normaliza por tamaño

    endpoints   = bwmorph(skel, 'endpoints');
    branchpoints = bwmorph(skel, 'branchpoints');
    nEnd = sum(endpoints(:));
    nBranch = sum(branchpoints(:));

    % 8) Fourier descriptors del contorno (robustos a traslación/escala, bastante estables)
    B = bwboundaries(maskR);
    if isempty(B)
        fd = zeros(1,4);
    else
        b = B{1};
        % convertir a señal compleja
        z = (b(:,2) + 1i*b(:,1));                 % x + i*y
        % re-muestrear a Nboundary puntos
        t = linspace(1, numel(z), Nboundary);
        z = interp1(1:numel(z), z, t, 'linear');

        % quitar traslación
        z = z - mean(z);

        % FFT
        Z = fft(z);

        % invariancia a escala: normalizar por el primer armónico (magnitud)
        den = max(abs(Z(2)), 1e-12);
        mag = abs(Z) / den;

        % coger magnitudes 3..6 (equiv a 2..5 si cuentas desde 1: DC es 1)
        % aquí usamos: mag(3), mag(4), mag(5), mag(6)
        idx = 3:(2+Kfourier); % 3..7 si Kfourier=5 -> luego nos quedamos 3..6 (4 feats)
        magSel = mag(idx);
        fd = magSel(1:4).'; % 4 features
    end

    % 9) Vector final (14 features)
    feat = double([ ...
        circ, aspect, ext, sol, conv, ecc, euler, ...
        skelLenNorm, nEnd, nBranch, ...
        fd(1), fd(2), fd(3), fd(4) ...
    ]);
end

%% Version anterior

% function feat = extractShapeFeatures(I)
% % EXTRACTSHAPEFEATURES  Features de forma (sin color) para piezas LEGO segmentadas.
% % Compatible con MATLAB R2022b (sin 'MomentsHu' en regionprops).
% % Devuelve 15 features: 8 clásicas + 7 Hu (log-escala).
% 
%     % ---- 1) Asegurar grayscale ----
%     if size(I,3) == 3
%         Ig = rgb2gray(I);
%     else
%         Ig = I;
%     end
%     Ig = im2double(Ig);
% 
%     % ---- 2) Máscara del objeto (fondo negro) ----
%     mask = Ig > 0.02;
%     mask = imfill(mask, 'holes');
%     mask = bwareaopen(mask, 200);
% 
%     if ~any(mask(:))
%         feat = zeros(1, 15);
%         return;
%     end
% 
%     % ---- 3) Componente más grande ----
%     CC = bwconncomp(mask, 8);
%     if CC.NumObjects > 1
%         areas = cellfun(@numel, CC.PixelIdxList);
%         [~, imax] = max(areas);
%         mask = false(size(mask));
%         mask(CC.PixelIdxList{imax}) = true;
%     end
% 
%     % ---- 4) Regionprops (sin MomentsHu) ----
%     S = regionprops(mask, ...
%         'Area','Perimeter','Eccentricity','Solidity','Extent', ...
%         'MajorAxisLength','MinorAxisLength','BoundingBox');
% 
%     A   = S.Area;
%     P   = S.Perimeter;
%     ecc = S.Eccentricity;
%     sol = S.Solidity;
%     ext = S.Extent;
% 
%     maj = S.MajorAxisLength;
%     mino = S.MinorAxisLength;
%     aspect = maj / max(mino, 1e-9);
% 
%     bb = S.BoundingBox;              % [x y w h]
%     bb_ratio = bb(3) / max(bb(4), 1e-9);
% 
%     % Circularidad
%     circ = (4*pi*A) / max(P^2, 1e-9);
% 
%     % ---- 5) Hu moments (7) calculados a mano desde la máscara ----
%     hu = huMomentsFromMask(mask);         % 1x7
% 
%     % Log-escala (recomendado)
%     hu_log = -sign(hu) .* log10(abs(hu) + 1e-12);
% 
%     % ---- 6) Vector final (15) ----
%     feat = double([A, P, circ, ecc, sol, ext, aspect, bb_ratio, hu_log]);
% end
% 
% 
% function hu = huMomentsFromMask(BW)
% % Calcula los 7 Hu moments clásicos a partir de una máscara binaria.
% % BW: logical
% 
%     BW = logical(BW);
% 
%     % Coordenadas de píxeles "on"
%     [y, x] = find(BW);
%     if isempty(x)
%         hu = zeros(1,7);
%         return;
%     end
% 
%     x = double(x);
%     y = double(y);
% 
%     % Centroide
%     xbar = mean(x);
%     ybar = mean(y);
% 
%     % Coordenadas centradas
%     xc = x - xbar;
%     yc = y - ybar;
% 
%     % Momentos centrales mu_pq
%     mu00 = numel(x); % suma de 1s
%     mu11 = sum(xc .* yc);
%     mu20 = sum(xc.^2);
%     mu02 = sum(yc.^2);
%     mu30 = sum(xc.^3);
%     mu03 = sum(yc.^3);
%     mu21 = sum(xc.^2 .* yc);
%     mu12 = sum(xc .* yc.^2);
% 
%     % Momentos centrales normalizados eta_pq
%     % eta_pq = mu_pq / mu00^(1 + (p+q)/2)
%     eta11 = mu11 / (mu00^(1 + (1+1)/2));
%     eta20 = mu20 / (mu00^(1 + (2+0)/2));
%     eta02 = mu02 / (mu00^(1 + (0+2)/2));
%     eta30 = mu30 / (mu00^(1 + (3+0)/2));
%     eta03 = mu03 / (mu00^(1 + (0+3)/2));
%     eta21 = mu21 / (mu00^(1 + (2+1)/2));
%     eta12 = mu12 / (mu00^(1 + (1+2)/2));
% 
%     % Hu invariants (7)
%     hu = zeros(1,7);
%     hu(1) = eta20 + eta02;
%     hu(2) = (eta20 - eta02)^2 + 4*eta11^2;
%     hu(3) = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2;
%     hu(4) = (eta30 + eta12)^2 + (eta21 + eta03)^2;
%     hu(5) = (eta30 - 3*eta12)*(eta30 + eta12)*((eta30 + eta12)^2 - 3*(eta21 + eta03)^2) + ...
%             (3*eta21 - eta03)*(eta21 + eta03)*(3*(eta30 + eta12)^2 - (eta21 + eta03)^2);
%     hu(6) = (eta20 - eta02)*((eta30 + eta12)^2 - (eta21 + eta03)^2) + ...
%             4*eta11*(eta30 + eta12)*(eta21 + eta03);
%     hu(7) = (3*eta21 - eta03)*(eta30 + eta12)*((eta30 + eta12)^2 - 3*(eta21 + eta03)^2) - ...
%             (eta30 - 3*eta12)*(eta21 + eta03)*(3*(eta30 + eta12)^2 - (eta21 + eta03)^2);
% end

