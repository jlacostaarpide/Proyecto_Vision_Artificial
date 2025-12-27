function z = extractBoundaryLego(I, N)
% Devuelve frontera como vector complejo z (Nx1), normalizada en centro y escala.
% I: imagen segmentada (fondo negro o casi negro funciona mejor)
% N: nº de puntos en la frontera (ej 256)

    if nargin < 2, N = 256; end
    if ~isfloat(I), I = im2double(I); end
    if size(I,3)==3, Ig = rgb2gray(I); else, Ig = I; end

    % Umbral simple para objeto vs fondo (ajusta si hace falta)
    bw = Ig > 0.03;
    bw = bwareaopen(bw, 150);
    bw = imfill(bw,'holes');

    % quedarnos con el componente más grande
    CC = bwconncomp(bw,8);
    if CC.NumObjects==0
        z = complex(zeros(N,1));
        return;
    end
    if CC.NumObjects>1
        areas = cellfun(@numel, CC.PixelIdxList);
        [~,imax] = max(areas);
        bw2 = false(size(bw));
        bw2(CC.PixelIdxList{imax}) = true;
        bw = bw2;
    end

    B = bwboundaries(bw);
    b = B{1};                 % [row col]
    x = b(:,2); y = b(:,1);   % col=x, row=y

    % resamplear a N puntos por longitud de arco
    dx = diff(x); dy = diff(y);
    s = [0; cumsum(sqrt(dx.^2 + dy.^2))];
    if s(end) == 0
        z = complex(zeros(N,1));
        return;
    end
    t = linspace(0, s(end), N).';
    xr = interp1(s, x, t, 'linear');
    yr = interp1(s, y, t, 'linear');

    z = complex(xr, yr);

    % normalizar traslación y escala
    z = z - mean(z);
    z = z / (sqrt(mean(abs(z).^2)) + 1e-12);
end
