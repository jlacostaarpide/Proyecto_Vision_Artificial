clear; clc;

% Códigos de clase válidos
validCodes = {'01'};

% Carpeta donde tienes las IMÁGENES YA SEGMENTADAS (una pieza por archivo)
testFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_local';

% Listar imágenes
filesJPG = dir(fullfile(testFolder, '*.jpg'));
filesPNG = dir(fullfile(testFolder, '*.png'));
files    = [filesJPG; filesPNG];

% Filtrar solo archivos cuyo nombre empiece por X
isValid = false(numel(files),1);

for i = 1:numel(files)
    [~, baseName, ~] = fileparts(files(i).name);
    partes = split(baseName, '_');
    if ~isempty(partes) && ismember(partes{1}, validCodes)
        isValid(i) = true;
    end
end

files = files(isValid);

fprintf('Tras filtrar por código: %d imágenes válidas\n', numel(files));

%% Filtrar solo orientación 315
isOri0 = false(numel(files),1);

for i = 1:numel(files)
    [~, baseName, ~] = fileparts(files(i).name);
    partes = split(baseName, '_');

    % Seguridad: comprobar que hay al menos 2 partes
    if numel(partes) >= 2 && strcmp(partes{2}, '315')
        isOri0(i) = true;
    end
end

files_ori0 = files(isOri0);

fprintf('De esas, orientación 315: %d imágenes\n', numel(files_ori0));

%% Construcción de plantilla 0º usando varias imágenes

N = numel(files_ori0);
if N == 0
    error('No hay imágenes de orientación 315');
end

templates = [];

for i = 1:N
    imgPath = fullfile(files_ori0(i).folder, files_ori0(i).name);
    I = imread(imgPath);

    mask = extractMaskLego(I);

    % Contorno
    B = bwboundaries(mask);
    if isempty(B)
        continue;
    end

    b = B{1};
    z = b(:,2) + 1i*b(:,1);

    % Re-muestreo
    Nboundary = 128;
    t = linspace(1, numel(z), Nboundary);
    z = interp1(1:numel(z), z, t, 'linear');

    % Normalizaciones
    z = z - mean(z);              % traslación
    Z = fft(z);
    Z = Z / max(abs(Z(2)),1e-12); % escala

    templates = [templates; Z.']; %#ok<AGROW>
end

% Plantilla final = media compleja
template01_315.Z = mean(templates,1);
template01_315.Nboundary = Nboundary;
template01_315.Kuse = 6:20;   % armónicos usados
template01_315.code = '01';
template01_315.orientation = 0;

save('template01_315.mat','template01_315');

fprintf('Plantilla creada con %d imágenes\n', size(templates,1));
