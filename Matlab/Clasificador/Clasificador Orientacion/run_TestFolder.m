%% run_test_from_list_or_folder.m
clear; clc;

% ===== CONFIG =====
code = "03";
tplFolder = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Orientacion\Templates";
resultsFolder = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Orientacion\Results";
anglesStr = ["000","045","090","135","180","225","270","315"];
pitchStr  = ["10","40","70","90"];

% ===== MODO DE TEST =====
% Elige UNO (deja el otro comentado con %)

%MODE = "FOLDER";   % <-- USA ESTA para carpeta nueva
MODE = "LIST";   % <-- USA ESTA para test_list_XX.mat

% Carpeta alternativa si MODE="FOLDER"
segFolder = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_test2_local";

%% ===== CARGAR PLANTILLAS =====
templatesCell = {};
for a = 1:numel(anglesStr)
    for p = 1:numel(pitchStr)
        ang = anglesStr(a);
        pit = pitchStr(p);

        f = fullfile(tplFolder, sprintf("tpl_%s_%s_%s.mat", code, ang, pit));
        if ~isfile(f), continue; end

        S = load(f);
        if ~isfield(S,"tpl"), continue; end

        templatesCell{end+1} = S.tpl; %#ok<SAGROW>
    end
end

if isempty(templatesCell)
    error("No hay plantillas tpl_%s_*.mat en %s", code, tplFolder);
end

% Unificar fields para struct array (robusto)
allFields = {};
for k = 1:numel(templatesCell)
    allFields = union(allFields, fieldnames(templatesCell{k}));
end
for k = 1:numel(templatesCell)
    for ff = 1:numel(allFields)
        fn = allFields{ff};
        if ~isfield(templatesCell{k}, fn)
            templatesCell{k}.(fn) = [];
        end
    end
end
templates = [templatesCell{:}];

fprintf("Cargadas %d plantillas.\n", numel(templates));

%% ===== CARGAR TEST ITEMS SEGÚN MODO =====
if MODE == "LIST"
    %%% ===== (A) MODO LISTA: usa test_list_XX.mat =====
    testListFile = fullfile(tplFolder, sprintf("test_list_%s.mat", code));
    if ~isfile(testListFile)
        error("No existe %s. Ejecuta antes buildTemplatesYawPitch_split.m", testListFile);
    end
    S = load(testListFile);
    if ~isfield(S,"testItems")
        error("En %s no existe la variable 'testItems'", testListFile);
    end
    testItems = S.testItems;

elseif MODE == "FOLDER"
    %%% ===== (B) MODO CARPETA: lee ficheros y saca GT del nombre =====
    files = [dir(fullfile(segFolder,'*.jpg')); dir(fullfile(segFolder,'*.png')); dir(fullfile(segFolder,'*.jpeg'))];
    if isempty(files), error("No hay imágenes en %s", segFolder); end

    tmp = struct('path',{},'gtYaw',{},'gtPitch',{});
    for i = 1:numel(files)
        fname = string(files(i).name);
        [~, base, ~] = fileparts(fname);
        parts = split(base,'_');

        % Esperado: CODE_YAW_PITCH_...
        if numel(parts) < 3, continue; end
        if string(parts{1}) ~= code, continue; end

        gtYaw   = str2double(parts{2});
        gtPitch = str2double(parts{3});
        if isnan(gtYaw) || isnan(gtPitch), continue; end

        tmp(end+1).path    = fullfile(files(i).folder, files(i).name); %#ok<SAGROW>
        tmp(end).gtYaw     = gtYaw;
        tmp(end).gtPitch   = gtPitch;
    end

    testItems = tmp;
else
    error("MODE debe ser 'LIST' o 'FOLDER'");
end

fprintf("Test items: %d\n\n", numel(testItems));
if isempty(testItems)
    error("No hay testItems válidos para code=%s en MODE=%s", code, MODE);
end

%% ===== ABRIR LOG ÚNICO =====
outTxt = fullfile(resultsFolder, sprintf("results_all_%s_%s.txt", code, lower(MODE)));
fid = fopen(outTxt, 'w');
if fid < 0
    error("No puedo abrir %s para escritura", outTxt);
end

fprintf(fid, "RESULTADOS TEST (code=%s, mode=%s)\n", code, MODE);
fprintf(fid, "=============================================\n\n");

%% ===== TEST =====
n = numel(testItems);
nOK_yaw  = 0;
nOK_both = 0;

fails = struct('file',{}, 'gtYaw',{}, 'gtPitch',{}, 'predYaw',{}, 'predPitch',{}, 'score',{}, 'gap',{});

for i = 1:n
    path    = testItems(i).path;
    gtYaw   = testItems(i).gtYaw;
    gtPitch = testItems(i).gtPitch;

    I = imread(path);

    [predYaw, predPitch, scores, gap] = predictYawPitch_byTemplate(I, templates);

    okYaw  = (predYaw == gtYaw);
    okBoth = okYaw && (predPitch == gtPitch);

    if okYaw,  nOK_yaw  = nOK_yaw  + 1; end
    if okBoth, nOK_both = nOK_both + 1; end

    tag = "OK"; if ~okYaw, tag = "FAIL"; end

    [~,name,ext] = fileparts(path);
    fname = string(name) + string(ext);

    line = sprintf("%-24s GT:%03d/%02d -> Pred:%03d/%02d (score=%.3f gap=%.3f) [%s]\n", ...
        char(fname), gtYaw, gtPitch, predYaw, predPitch, max(scores), gap, char(tag));

    fprintf("%s", line);
    fprintf(fid, "%s", line);

    if ~okYaw
        fails(end+1).file = char(fname); %#ok<SAGROW>
        fails(end).gtYaw = gtYaw;
        fails(end).gtPitch = gtPitch;
        fails(end).predYaw = predYaw;
        fails(end).predPitch = predPitch;
        fails(end).score = max(scores);
        fails(end).gap = gap;
    end
end

accYaw  = 100 * nOK_yaw  / max(n,1);
accBoth = 100 * nOK_both / max(n,1);

%% ===== RESUMEN =====
fprintf(fid, "\n=========== RESUMEN TEST ===========\n");
fprintf(fid, "N test:            %d\n", n);
fprintf(fid, "Acc YAW:           %.2f %% (%d)\n", accYaw, nOK_yaw);
fprintf(fid, "Acc YAW+PITCH:     %.2f %% (%d)\n", accBoth, nOK_both);
fprintf(fid, "===================================\n");

fprintf(fid, "\n=========== FALLOS (YAW) ===========\n");
if isempty(fails)
    fprintf(fid, "Ninguno.\n");
else
    for j = 1:numel(fails)
        r = fails(j);
        fprintf(fid, "%-24s GT:%03d/%02d -> Pred:%03d/%02d (score=%.3f gap=%.3f)\n", ...
            r.file, r.gtYaw, r.gtPitch, r.predYaw, r.predPitch, r.score, r.gap);
    end
end
fprintf(fid, "===================================\n");

fclose(fid);

fprintf("\nGuardado log completo en: %s\n", outTxt);
