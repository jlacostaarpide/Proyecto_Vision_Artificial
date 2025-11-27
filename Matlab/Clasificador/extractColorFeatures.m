function feat = extractColorFeatures(I)
    % Entrada: I = imagen RGB de una pieza con fondo negro
    % Salida: feat = vector fila con características de color (6 valores)

    I = im2double(I);
    if size(I,3) == 1
        I = repmat(I,[1 1 3]);
    end

    % Pasar a HSV
    I_hsv = rgb2hsv(I);
    H = I_hsv(:,:,1);
    S = I_hsv(:,:,2);
    V = I_hsv(:,:,3);

    % Como ya tienes fondo negro, nos vale todo el píxel != 0
    mask = any(I > 0, 3);    % píxeles que no son fondo
    H = H(mask);
    S = S(mask);
    V = V(mask);

    % Estadísticos simples
    feat = [mean(H) std(H) ...
            mean(S) std(S) ...
            mean(V) std(V)];
end
