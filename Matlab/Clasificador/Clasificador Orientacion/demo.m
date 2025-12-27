clear; clc;

% 1) cargar plantilla (la creas una vez)
% template0 = buildOrientationTemplate("C:\...\07_000_70_003.jpg","template07.mat");
load("template07.mat","template0");

% 2) carpeta a probar
folderPath = "C:\Users\...\carpeta_orientaciones_07";
files = [dir(fullfile(folderPath,'*.jpg')); dir(fullfile(folderPath,'*.png'))];

% ordenar numérico si quieres, aquí no es clave
for i = 1:numel(files)
    imgPath = fullfile(files(i).folder, files(i).name);
    I = imread(imgPath);

    [angQ, scores] = predictOrientation45(I, template0);

    fprintf("%s -> ang=%d (scores max=%.3f)\n", files(i).name, angQ, max(scores));
end
