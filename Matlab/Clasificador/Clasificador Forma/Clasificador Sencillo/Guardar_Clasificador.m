%% train_9v12_final.m
clear; clc;

load('legoFeatures_TRAIN_shape_24carac_F_amarillas.mat'); % trae F_a
T = F_a;

feat6 = { ...
 'AreaNorm','PerimNorm','ProjH_entropy', ...
 'GridOccFrac_3x3','GridOccGini_3x3','GridOccDiagDiff_3x3' ...
};

% Asegurar solo 09 y 12
T.Label = categorical(T.Label);
T = T(ismember(string(T.Label), ["09","12"]), :);
T.Label = removecats(T.Label);

% Entrenar árbol base
t0 = fitctree(T(:,feat6), T.Label, ...
    "SplitCriterion","gdi", ...
    "MinLeafSize",3, ...
    "MinParentSize",6, ...
    "Surrogate","off");

% Poda: elegir nivel por CV (aquí 5-fold)
% (usa ClassificationTree/cvloss con nombre correcto: "KFold")
[~,~,~,bestLevel] = cvloss(t0, "SubTrees","all", "KFold",5);

tFinal = prune(t0, "Level", bestLevel);

% Guardar modelo + nombres + (opcional) opts de extracción
model = struct();
model.tree = tFinal;
model.featNames = feat6;

save("tree_9v12_final.mat","model");

fprintf("Modelo guardado: tree_9v12_final.mat (bestLevel=%d)\n", bestLevel);
