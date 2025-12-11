function feat = extractColorFeatures(I)
% EXTRACTCOLORFEATURES  Extrae características de color robustas de una pieza LEGO
%   I:  imagen RGB de la pieza con fondo negro (uint8 o double)
%   feat: vector fila de características (1x8)
%
%   [1] H_mean_circ   (media circular de H)
%   [2] H_var_circ    (varianza circular de H)
%   [3] S_median
%   [4] S_IQR
%   [5] V_median
%   [6] V_IQR
%   [7] S_mean
%   [8] V_mean

    % Asegurar double y 3 canales
    I = im2double(I);
    if size(I,3) == 1
        I = repmat(I,[1 1 3]);
    end

    %-------------------------------------------------------------
    % 1) Corrección (suave) de iluminación en espacio Lab
    %-------------------------------------------------------------
    Ilab = rgb2lab(I);          % L en [0,100], a,b ~ [-128,128]
    L = Ilab(:,:,1);
    A = Ilab(:,:,2);
    B = Ilab(:,:,3);

    % Ecualización adaptativa solo de L (contraste / iluminación)
    Lnorm = adapthisteq(L/100); % [0,1]
    Lcorr = Lnorm * 100;        % volver a [0,100]

    Ilab2 = cat(3, Lcorr, A, B);
    I_corr = lab2rgb(Ilab2);

    % Clip por si algún valor se va un poco
    I_corr = min(max(I_corr,0),1);

    %-------------------------------------------------------------
    % 2) Pasar a HSV y quedarnos solo con la pieza (no fondo)
    %-------------------------------------------------------------
    hsv = rgb2hsv(I_corr);
    H = hsv(:,:,1);
    S = hsv(:,:,2);
    V = hsv(:,:,3);

    % Máscara de pieza: píxeles no negros y con algo de valor
    mask = any(I > 0, 3) & (V > 0.05);   % usa la imagen original para el fondo

    H = H(mask);
    S = S(mask);
    V = V(mask);

    if isempty(H)
        % Por seguridad: si algo falla y no hay píxeles válidos
        feat = zeros(1,8);
        return;
    end

    %-------------------------------------------------------------
    % 3) Estadísticos robustos
    %-------------------------------------------------------------

    % --- Hue: media y varianza CIRCULAR (Hue es angular) ---
    % H está en [0,1] → ángulo = 2*pi*H
    ang = 2*pi*H;
    z = exp(1j * ang);
    R = mean(z);               % media de vectores unitarios
    H_mean_circ = angle(R) / (2*pi);   % media circular normalizada [0,1]
    if H_mean_circ < 0
        H_mean_circ = H_mean_circ + 1; % asegurar [0,1]
    end

    % Varianza circular (0 = todos iguales, 1 = muy disperso)
    H_var_circ = 1 - abs(R);

    % --- Saturación y Valor: mediana e IQR + medias ---
    S_median = median(S);
    S_IQR    = iqr(S);
    V_median = median(V);
    V_IQR    = iqr(V);

    S_mean   = mean(S);
    V_mean   = mean(V);

    % Vector final de características
    feat = [H_mean_circ, H_var_circ, ...
            S_median, S_IQR, ...
            V_median, V_IQR, ...
            S_mean, V_mean];
end
