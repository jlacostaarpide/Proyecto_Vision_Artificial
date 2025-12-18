function [feat, featNames] = extractColorShapeFeatures(I)
% EXTRACTCOLORSHAPEFEATURES  Features combinadas (color + forma) para LEGO segmentado.
%
% Entrada:
%   I : imagen RGB (o gray) de la pieza segmentada con fondo negro.
%
% Salida:
%   feat      : 1x22 double  (8 color + 14 shape)
%   featNames : 1x22 cellstr (nombres en el mismo orden)
%
% Orden:
%   Color (8):
%     1 H_mean_circ
%     2 H_var_circ
%     3 S_median
%     4 S_IQR
%     5 V_median
%     6 V_IQR
%     7 S_mean
%     8 V_mean
%   Shape (14):
%     9  Circularity
%     10 AspectRatio
%     11 Extent
%     12 Solidity
%     13 Convexity
%     14 Eccentricity
%     15 EulerNumber
%     16 SkelLenNorm
%     17 SkelEndpoints
%     18 SkelBranchpoints
%     19 FD2
%     20 FD3
%     21 FD4
%     22 FD5

    % =======================
    % 0) Asegurar formato
    % =======================
    I = im2double(I);
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end

    % =======================
    % 1) FEATURES DE COLOR (8)
    % =======================
    featColor = local_extractColorFeatures(I);

    % =======================
    % 2) FEATURES DE FORMA (14)
    % =======================
    featShape = local_extractShapeFeatures(I);

    % =======================
    % 3) CONCATENAR
    % =======================
    feat = [featColor, featShape];

    featNames = { ...
        'H_mean_circ','H_var_circ','S_median','S_IQR','V_median','V_IQR','S_mean','V_mean', ...
        'Circularity','AspectRatio','Extent','Solidity','Convexity','Eccentricity','EulerNumber', ...
        'SkelLenNorm','SkelEndpoints','SkelBranchpoints','FD2','FD3','FD4','FD5' ...
    };
end

% ======================================================================
% =========================  COLOR (8)  ================================
% ======================================================================
function feat = local_extractColorFeatures(I)
% Versión basada en tu extractColorFeatures, tal cual la lógica.

    % 1) Corrección suave de iluminación en Lab
    Ilab = rgb2lab(I);
    L = Ilab(:,:,1);
    A = Ilab(:,:,2);
    B = Ilab(:,:,3);

    Lnorm = adapthisteq(L/100);
    Lcorr = Lnorm * 100;

    Ilab2 = cat(3, Lcorr, A, B);
    I_corr = lab2rgb(Ilab2);
    I_corr = min(max(I_corr,0),1);

    % 2) HSV + máscara de pieza
    hsvI = rgb2hsv(I_corr);
    H = hsvI(:,:,1);
    S = hsvI(:,:,2);
    V = hsvI(:,:,3);

    % máscara: no negro + algo de valor (usando original para fondo)
    mask = any(I > 0, 3) & (V > 0.05);

    H = H(mask);
    S = S(mask);
    V = V(mask);

    if isempty(H)
        feat = zeros(1,8);
        return;
    end

    % 3) Estadísticos robustos
    ang = 2*pi*H;
    z = exp(1j * ang);
    R = mean(z);
    H_mean_circ = angle(R) / (2*pi);
    if H_mean_circ < 0, H_mean_circ = H_mean_circ + 1; end
    H_var_circ = 1 - abs(R);

    S_median = median(S);
    S_IQR    = iqr(S);
    V_median = median(V);
    V_IQR    = iqr(V);

    S_mean   = mean(S);
    V_mean   = mean(V);

    feat = [H_mean_circ, H_var_circ, S_median, S_IQR, V_median, V_IQR, S_mean, V_mean];
end

% ======================================================================
% =========================  SHAPE (14)  ===============================
% ======================================================================
function feat = local_extractShapeFeatures(I)
% Versión robusta de forma (la de 14 features que estabas usando)

    % Parámetros
    tBlackMin        = 0.03;
    minObjArea       = 300;
    holeSmallMaxArea = 200;
    closeRadius      = 3;
    openRadius       = 2;
    Nboundary        = 128;
    Kfourier         = 5;

    % 1) Grayscale
    Ig = rgb2gray(I);

    % 2) Máscara "no negro"
    mask = Ig > tBlackMin;

    % 3) Limpieza
    mask = bwareaopen(mask, minObjArea);
    mask = imclose(mask, strel('disk', closeRadius));
    mask = imopen(mask,  strel('disk', openRadius));

    % 4) Rellenar solo agujeros pequeños (conservar grandes)
    maskFilled = imfill(mask, 'holes');
    holes = maskFilled & ~mask;
    holesSmall = bwareaopen(holes, holeSmallMaxArea); % se quedan grandes
    holesToFill = holes & ~holesSmall;                % los pequeños
    mask = mask | holesToFill;

    % 5) Componente más grande
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

    % 6) Normalizar orientación
    S0 = regionprops(mask, 'Orientation', 'BoundingBox');
    ang = -S0.Orientation;
    maskR = imrotate(mask, ang, 'nearest', 'loose');

    S1 = regionprops(maskR, 'BoundingBox');
    bb = S1.BoundingBox;
    maskR = imcrop(maskR, bb);

    % 7) props
    S = regionprops(maskR, 'Area','Perimeter','Eccentricity','Solidity','Extent', ...
                          'MajorAxisLength','MinorAxisLength','ConvexArea','EulerNumber');

    A  = S.Area;
    P  = max(S.Perimeter, 1e-9);

    circ = (4*pi*A) / (P^2);
    ecc  = S.Eccentricity;
    sol  = S.Solidity;
    ext  = S.Extent;

    maj = max(S.MajorAxisLength, 1e-9);
    mino = max(S.MinorAxisLength, 1e-9);
    aspect = maj / mino;

    conv = A / max(S.ConvexArea, 1e-9);
    euler = S.EulerNumber;

    % 8) Skeleton
    skel = bwmorph(maskR, 'skel', Inf);
    skelLen = sum(skel(:));
    skelLenNorm = skelLen / max(sqrt(A), 1e-9);

    endpoints = bwmorph(skel, 'endpoints');
    branchpoints = bwmorph(skel, 'branchpoints');
    nEnd = sum(endpoints(:));
    nBranch = sum(branchpoints(:));

    % 9) Fourier descriptors del contorno
    B = bwboundaries(maskR);
    if isempty(B)
        fd = zeros(1,4);
    else
        b = B{1};
        z = (b(:,2) + 1i*b(:,1));
        t = linspace(1, numel(z), Nboundary);
        z = interp1(1:numel(z), z, t, 'linear');
        z = z - mean(z);

        Z = fft(z);
        den = max(abs(Z(2)), 1e-12);
        mag = abs(Z) / den;

        idx = 3:(2+Kfourier);     % 3..7 (si Kfourier=5)
        magSel = mag(idx);
        fd = magSel(1:4).';       % FD2..FD5
    end

    feat = double([ ...
        circ, aspect, ext, sol, conv, ecc, euler, ...
        skelLenNorm, nEnd, nBranch, ...
        fd(1), fd(2), fd(3), fd(4) ...
    ]);
end
