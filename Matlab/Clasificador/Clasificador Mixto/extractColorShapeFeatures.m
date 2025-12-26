function [feat, featNames] = extractColorShapeFeatures(I)
% EXTRACTCOLORSHAPEFEATURES  Features reducidas (color + forma) para LEGO segmentado.
%
% Basado en tu ranking combinado, nos quedamos con las que más aportan:
%   Extent, Solidity, V_mean, Eccentricity, SkelLenNorm, Circularity,
%   H_mean_circ, S_mean, V_IQR, S_median, FD5, EulerNumber
%
% Entrada:
%   I : imagen RGB (o gray) de la pieza segmentada con fondo negro.
%
% Salida:
%   feat      : 1x12 double
%   featNames : 1x12 cellstr (nombres en el mismo orden)

    % =======================
    % 0) Asegurar formato
    % =======================
    I = im2double(I);
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end

    % =======================
    % 1) Features completas (como antes)
    % =======================
    featColorFull = local_extractColorFeatures(I);  % 1x8
    featShapeFull = local_extractShapeFeatures(I);  % 1x14

    % Color full (5):
    % [H_mean_circ, H_var_circ, S_median, S_IQR, V_median, V_IQR, S_mean, V_mean]
    H_mean_circ = featColorFull(1);
    S_median    = featColorFull(3);
    V_IQR       = featColorFull(6);
    S_mean      = featColorFull(7);
    V_mean      = featColorFull(8);

    % Shape full (7) en el ORDEN de tu función actual:
    % [circ, aspect, ext, sol, conv, ecc, euler, skelLenNorm, nEnd, nBranch, FD2, FD3, FD4, FD5]
    Circularity   = featShapeFull(1);
    Extent        = featShapeFull(3);
    Solidity      = featShapeFull(4);
    Eccentricity  = featShapeFull(6);
    EulerNumber   = featShapeFull(7);
    SkelLenNorm   = featShapeFull(8);
    FD5           = featShapeFull(14);

    % =======================
    % 2) Selección final (1x12) en un orden coherente
    % =======================
    feat = double([ ...
        Extent, Solidity, V_mean, Eccentricity, SkelLenNorm, Circularity, ...
        H_mean_circ, S_mean, V_IQR, S_median, FD5, EulerNumber ...
    ]);

    featNames = { ...
        'Extent','Solidity','V_mean','Eccentricity','SkelLenNorm','Circularity', ...
        'H_mean_circ','S_mean','V_IQR','S_median','FD5','EulerNumber' ...
    };
end

% ======================================================================
% =========================  COLOR (8)  ================================
% ======================================================================
function feat = local_extractColorFeatures(I)
% Igual que tu lógica (8 features)

    Ilab = rgb2lab(I);
    L = Ilab(:,:,1);
    A = Ilab(:,:,2);
    B = Ilab(:,:,3);

    Lnorm = adapthisteq(L/100);
    Lcorr = Lnorm * 100;

    Ilab2 = cat(3, Lcorr, A, B);
    I_corr = lab2rgb(Ilab2);
    I_corr = min(max(I_corr,0),1);

    hsvI = rgb2hsv(I_corr);
    H = hsvI(:,:,1);
    S = hsvI(:,:,2);
    V = hsvI(:,:,3);

    mask = any(I > 0, 3) & (V > 0.05);

    H = H(mask);
    S = S(mask);
    V = V(mask);

    if isempty(H)
        feat = zeros(1,8);
        return;
    end

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
% Igual que tu lógica (14 features)

    tBlackMin        = 0.03;
    minObjArea       = 300;
    holeSmallMaxArea = 200;
    closeRadius      = 3;
    openRadius       = 2;
    Nboundary        = 128;
    Kfourier         = 5;

    Ig = rgb2gray(I);
    mask = Ig > tBlackMin;

    mask = bwareaopen(mask, minObjArea);
    mask = imclose(mask, strel('disk', closeRadius));
    mask = imopen(mask,  strel('disk', openRadius));

    maskFilled = imfill(mask, 'holes');
    holes = maskFilled & ~mask;
    holesSmall = bwareaopen(holes, holeSmallMaxArea);
    holesToFill = holes & ~holesSmall;
    mask = mask | holesToFill;

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

    S0 = regionprops(mask, 'Orientation', 'BoundingBox');
    ang = -S0.Orientation;
    maskR = imrotate(mask, ang, 'nearest', 'loose');

    S1 = regionprops(maskR, 'BoundingBox');
    bb = S1.BoundingBox;
    maskR = imcrop(maskR, bb);

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

    skel = bwmorph(maskR, 'skel', Inf);
    skelLen = sum(skel(:));
    skelLenNorm = skelLen / max(sqrt(A), 1e-9);

    endpoints = bwmorph(skel, 'endpoints');
    branchpoints = bwmorph(skel, 'branchpoints');
    nEnd = sum(endpoints(:));
    nBranch = sum(branchpoints(:));

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

        idx = 3:(2+Kfourier);
        magSel = mag(idx);
        fd = magSel(1:4).';   % FD2..FD5
    end

    feat = double([ ...
        circ, aspect, ext, sol, conv, ecc, euler, ...
        skelLenNorm, nEnd, nBranch, ...
        fd(1), fd(2), fd(3), fd(4) ...
    ]);
end
