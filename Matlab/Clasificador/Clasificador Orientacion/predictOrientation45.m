function [angBest, scores] = predictOrientation45(I, templates, anglesNum, N)
N = 256; % o 128 o 256 pero fijo
% templates{k}.Z debe existir (vector complejo)
    if nargin < 3 || isempty(anglesNum), anglesNum = 0:45:315; end
    if nargin < 4, N = templates{1}.Nboundary; end

    z = extractBoundaryLego(I, N);
    K = numel(templates);
    scores = zeros(K,1);

    for k = 1:K
        zt = templates{k}.Z;

        % correlación circular para alinear punto inicial
        c = ifft(fft(z).*conj(fft(zt)));
        [mx, ~] = max(real(c));

        % score: más alto = más parecido
        scores(k) = mx;  % ya está bastante bien normalizado por extractBoundaryLego
    end

    [~, idx] = max(scores);
    angBest = anglesNum(idx);
end
