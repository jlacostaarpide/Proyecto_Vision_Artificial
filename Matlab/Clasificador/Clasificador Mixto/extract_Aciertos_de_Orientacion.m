%% eval_yaw_from_log.m
% Lee resultados_Mtest_dobleClassificador.txt y evalúa si ORI yaw coincide
% con el yaw que aparece en el nombre del fichero (CODE_YAW_PITCH_...)

clear; clc;

logFile = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\resultados_Mtest_dobleClassificador.txt";

if ~isfile(logFile)
    error("No existe el archivo: %s", logFile);
end

fid = fopen(logFile, 'r');
if fid < 0
    error("No puedo abrir: %s", logFile);
end

nLines = 0;
nParsed = 0;
nOK = 0;
nFAIL = 0;
nNoORI = 0;
nBadName = 0;

fails = strings(0,1);

while true
    tline = fgetl(fid);
    if ~ischar(tline), break; end
    nLines = nLines + 1;

    % Solo nos interesan líneas que contienen ORI=
    if ~contains(tline, "ORI=")
        continue;
    end
    nParsed = nParsed + 1;

    % 1) Extraer ORI yaw (primero de ORI=YYY/PP)
    % Ej: " ... ORI=225/70 (s=... g=...) | ..."
    tokOri = regexp(tline, 'ORI=(\d{1,3})/(\d{1,2})', 'tokens', 'once');
    if isempty(tokOri)
        nNoORI = nNoORI + 1;
        continue;
    end
    predYaw = str2double(tokOri{1});

    % 2) Extraer nombre de archivo al principio de la línea:
    % "[1887/1923] 12_225_70_003.jpg | REAL=..."
    tokName = regexp(tline, '\]\s+([^\s]+)\s+\|', 'tokens', 'once');
    if isempty(tokName)
        nBadName = nBadName + 1;
        continue;
    end
    fname = string(tokName{1});  % "12_225_70_003.jpg"

    % 3) Extraer yaw “GT” del nombre (segundo token tras split por "_")
    % Esperado: CODE_YAW_PITCH_...
    [~, base, ~] = fileparts(fname);
    parts = split(base, "_");
    if numel(parts) < 3
        nBadName = nBadName + 1;
        continue;
    end

    nameYaw = str2double(parts{2});  % "225" -> 225
    if isnan(nameYaw)
        nBadName = nBadName + 1;
        continue;
    end

    % 4) Comparar
    if predYaw == nameYaw
        nOK = nOK + 1;
    else
        nFAIL = nFAIL + 1;
        fails(end+1,1) = sprintf("%s | nameYaw=%03d  predYaw=%03d", fname, nameYaw, predYaw);
    end
end

fclose(fid);

nTotalEval = nOK + nFAIL;
acc = 0;
if nTotalEval > 0
    acc = 100 * (nOK / nTotalEval);
end

fprintf("\n===== EVAL ORI YAW vs YAW en NOMBRE =====\n");
fprintf("Archivo: %s\n", logFile);
fprintf("Líneas totales:                  %d\n", nLines);
fprintf("Líneas con ORI (intentadas):     %d\n", nParsed);
fprintf("Evaluadas correctamente:         %d\n", nTotalEval);
fprintf("Aciertos:                        %d\n", nOK);
fprintf("Fallos:                          %d\n", nFAIL);
fprintf("Accuracy YAW (solo evaluadas):   %.2f %%\n", acc);
fprintf("No parsea ORI:                   %d\n", nNoORI);
fprintf("Nombre raro/no parseable:        %d\n", nBadName);

% (Opcional) imprimir fallos
if ~isempty(fails)
    fprintf("\n--- FALLOS (primeros 50) ---\n");
    for i = 1:min(50, numel(fails))
        fprintf("%s\n", fails(i));
    end
end

% (Opcional) guardar un resumen a txt
outFile = replace(logFile, ".txt", "_EVAL_YAW.txt");
fid2 = fopen(outFile, 'w');
if fid2 >= 0
    fprintf(fid2, "EVAL ORI YAW vs YAW EN NOMBRE\n");
    fprintf(fid2, "Archivo: %s\n\n", logFile);
    fprintf(fid2, "Líneas totales:                %d\n", nLines);
    fprintf(fid2, "Líneas con ORI (intentadas):   %d\n", nParsed);
    fprintf(fid2, "Evaluadas:                     %d\n", nTotalEval);
    fprintf(fid2, "Aciertos:                      %d\n", nOK);
    fprintf(fid2, "Fallos:                        %d\n", nFAIL);
    fprintf(fid2, "Accuracy YAW:                  %.2f %%\n", acc);
    fprintf(fid2, "No parsea ORI:                 %d\n", nNoORI);
    fprintf(fid2, "Nombre raro/no parseable:      %d\n\n", nBadName);

    fprintf(fid2, "FALLOS:\n");
    if isempty(fails)
        fprintf(fid2, "Ninguno.\n");
    else
        for i = 1:numel(fails)
            fprintf(fid2, "%s\n", fails(i));
        end
    end
    fclose(fid2);
    fprintf("\nGuardado resumen en: %s\n", outFile);
end
