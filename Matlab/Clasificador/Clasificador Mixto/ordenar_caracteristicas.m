%% ================================================================
%% RANKING DE CARACTERÍSTICAS (4 métodos) + ranking combinado
%% Requiere: dataset con X (NxD) y G (Nx1) o tabla con Label
%% ================================================================

clear; clc;

% === 1) Carga tu dataset de características ===
% Cambia el nombre del .mat según el tuyo:
load("legoFeatures_TRAIN_color_shape_22carac.mat");  % <-- AJUSTA

% Esperado: o bien X,G; o bien una tabla (por ejemplo M) con Label.
% Adapta estas 3 líneas a tu caso real:
if exist("X","var") && exist("G","var")
    Xmat = X;
    Gvec = G;
    featNames = [];
elseif exist("M","var") && istable(M)
    % si tu tabla tiene 22 features + Label:
    Gvec = M.Label;
    Xmat = M{:, setdiff(M.Properties.VariableNames, {'Label','FileName'})};
    featNames = setdiff(M.Properties.VariableNames, {'Label','FileName'}, 'stable');
else
    error("No encuentro X,G ni una tabla M. Ajusta el bloque de carga.");
end

% Si no hay nombres, crea genéricos
D = size(Xmat,2);
if isempty(featNames)
    featNames = arrayfun(@(k)sprintf("F%02d",k), 1:D, "UniformOutput", false);
end

% Asegurar tipos correctos
Xmat = double(Xmat);
Gvec = Gvec(:);

% Si G es numeric, pásalo a categorical para algunos métodos
if ~iscategorical(Gvec)
    Gcat = categorical(Gvec);
else
    Gcat = Gvec;
end

%% === 2) MÉTODO 1: Chi-cuadrado (filter) ===
idx_chi2 = fscchi2(Xmat, Gcat);              % devuelve índices ordenados
rank_chi2 = zeros(D,1); rank_chi2(idx_chi2) = 1:D;

%% === 3) MÉTODO 2: mRMR (filter) ===
idx_mrmr = fscmrmr(Xmat, Gcat);
rank_mrmr = zeros(D,1); rank_mrmr(idx_mrmr) = 1:D;

%% === 4) MÉTODO 3: ReliefF (wrapper/heurístico) ===
% k = 10 vecinos como en apuntes (ajusta si quieres)
[idx_relief, w_relief] = relieff(Xmat, Gcat, 10);
rank_relief = zeros(D,1); rank_relief(idx_relief) = 1:D;

%% === 5) MÉTODO 4: Importancia de un árbol (wrapper) ===
% OJO: el árbol capta no-linealidades e interacciones.
tree = fitctree(Xmat, Gcat);
imp_tree = predictorImportance(tree);         % importancia por predictor
[~, idx_tree] = sort(imp_tree, 'descend');
rank_tree = zeros(D,1); rank_tree(idx_tree) = 1:D;

%% === 6) Ranking combinado (media de rangos) ===
rank_mean = mean([rank_chi2, rank_mrmr, rank_relief, rank_tree], 2);
[~, idx_comb] = sort(rank_mean, 'ascend');   % 1 = mejor

%% === 7) Tabla resumen (TOP y BOTTOM) ===
Tsum = table( ...
    string(featNames(:)), ...
    rank_chi2, rank_mrmr, rank_relief, rank_tree, ...
    rank_mean, imp_tree(:), w_relief(:), ...
    'VariableNames', {'Feature','Rank_Chi2','Rank_mRMR','Rank_ReliefF','Rank_Tree','Rank_Mean','TreeImportance','ReliefFWeight'} );

Tsum = sortrows(Tsum, 'Rank_Mean', 'ascend');

disp("=== TOP 10 características (según ranking combinado) ===");
disp(Tsum(1:min(10,height(Tsum)), :));

disp("=== BOTTOM 10 características (posibles candidatas a sobrar) ===");
disp(Tsum(max(1,height(Tsum)-9):height(Tsum), :));

%orden total:
disp("=== ORDEN FINAL ===");
disp(Tsum);

% (Opcional) guardar a CSV
writetable(Tsum, "ranking_features.csv");
fprintf("\nGuardado ranking_features.csv\n");
