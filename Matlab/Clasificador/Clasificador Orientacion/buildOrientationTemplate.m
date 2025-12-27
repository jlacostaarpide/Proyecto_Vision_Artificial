function template = buildOrientationTemplate(files, N)
% Guarda una plantilla basada en frontera compleja resampleada (sin máscara).
% template.zRef: frontera de referencia (Nx1 complex)

    if nargin < 2, N = 256; end

    Z = complex(zeros(N, numel(files)));
    keep = false(numel(files),1);

    for i = 1:numel(files)
        I = imread(fullfile(files(i).folder, files(i).name));
        z = extractBoundaryLego(I, N);

        if all(z==0), continue; end
        Z(:,i) = z;
        keep(i) = true;
    end

    Z = Z(:,keep);
    if isempty(Z)
        template.zRef = complex(zeros(N,1));
        template.N = N;
        template.n = 0;
        return;
    end

    % Elegir medoide: el que minimiza distancia media tras alineación circular
    M = size(Z,2);
    dmean = zeros(M,1);
    for i = 1:M
        di = 0;
        for j = 1:M
            di = di + minCircularDist(Z(:,i), Z(:,j));
        end
        dmean(i) = di / M;
    end
    [~, idx] = min(dmean);

    template.zRef = Z(:,idx);
    template.N = N;
    template.n = M;
end

function d = minCircularDist(z1, z2)
% distancia mínima permitiendo circular shift (para resolver punto inicial)
    c = ifft(fft(z1).*conj(fft(z2))); % correlación circular compleja
    [~,k] = max(real(c));            % mejor shift
    z2s = circshift(z2, k-1);
    d = mean(abs(z1 - z2s));
end
