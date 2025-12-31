%% train_9v12_LOO.m
clear; clc;

load('legoFeatures_TRAIN_shape_24carac_F_amarillas.mat'); % trae F_a
T = F_a;

featNames = { ...
 'AreaNorm','PerimNorm','ProjH_entropy', ...
 'GridOccFrac_3x3','GridOccGini_3x3','GridOccDiagDiff_3x3' ...
};

% Asegurar que Label es categorical y quedarnos solo con 09 y 12 (por si hay algo más)
T.Label = categorical(T.Label);
keep = ismember(string(T.Label), ["09","12"]);
T = T(keep, :);

X = T{:, featNames};   % <-- NUMÉRICO (N x 6)
Y = categorical(T.Label);
Y = removecats(Y);     % por si hay categorías vacías

% Orden fijo para confusion matrix
order = categorical(["09","12"]);

% LOO
cv = cvpartition(Y, "LeaveOut");

% =========================
% 1) ÁRBOL (portable a C)
% =========================
Yhat_tree = categorical(repmat(order(1), height(T), 1));  % init sin categoría rara
for k = 1:cv.NumTestSets
    tr = training(cv,k);
    te = test(cv,k);

    mdl = fitctree(T(tr, featNames), Y(tr), ...
        "MaxNumSplits", 12, ...
        "MinLeafSize",  5, ...
        "SplitCriterion","gdi");

    Yhat_tree(te) = predict(mdl, T(te, featNames));
end

acc_tree = mean(Yhat_tree == Y);
C_tree = confusionmat(Y, Yhat_tree, "Order", order);

fprintf("\nÁRBOL (LOO): Accuracy = %.2f %%\n", 100*acc_tree);
disp("Confusion matrix (rows=real, cols=pred) [09 12]:");
disp(C_tree);

% %% =========================
% % 2) NAIVE BAYES (portable: medias/vars por clase)
% %    OJO: zscore por fold (mejor, evita leakage)
% % =========================
% Yhat_nb = categorical(strings(height(T),1));
% for k = 1:cv.NumTestSets
%     tr = training(cv,k);
%     te = test(cv,k);
% 
%     Xtr = X(tr,:);
%     Xte = X(te,:);
%     Ytr = Y(tr);
% 
%     % --- Estandarizar con estadísticos del TRAIN (sin leakage) ---
%     mu = mean(Xtr,1);
%     sg = std(Xtr,0,1);
%     sg(sg < 1e-12) = 1;             % evita división por 0 global
% 
%     Xtrz = (Xtr - mu) ./ sg;
%     Xtez = (Xte - mu) ./ sg;
% 
%     % --- Quitar predictores con varianza 0 POR CLASE en el TRAIN ---
%     cats = categories(Ytr);
%     bad = false(1, size(Xtrz,2));
%     for c = 1:numel(cats)
%         idxc = (Ytr == cats{c});
%         if nnz(idxc) >= 2
%             v = var(Xtrz(idxc,:), 0, 1);
%             bad = bad | (v < 1e-12);
%         end
%     end
%     keep = ~bad;
% 
%     % entrenar NB gaussiano con columnas válidas
%     mdl = fitcnb(Xtrz(:,keep), Ytr, "DistributionNames","normal");
%     Yhat_nb(te) = predict(mdl, Xtez(:,keep));
% end
% 
% acc_nb = mean(Yhat_nb == Y);
% C_nb = confusionmat(Y, Yhat_nb, "Order", categories(Y));
% 
% fprintf("\nNAIVE BAYES (LOO): Accuracy = %.2f %%\n", 100*acc_nb);
% disp("Confusion matrix (rows=real, cols=pred) [09 12]:");
% disp(C_nb);

%%

treeFinal = fitctree(T(:,featNames), Y, ...
    'SplitCriterion','gdi', ...
    'MinLeafSize',3, ...
    'Surrogate','off');

imp = predictorImportance(treeFinal);   % 1x6
[impS, idx] = sort(imp, 'descend');

disp(table(string(featNames(idx))', impS', ...
    'VariableNames', {'Feature','Importance'}));
% bar(impS); grid on;
% set(gca,'XTick',1:numel(idx),'XTickLabel',featNames(idx),'XTickLabelRotation',45);
% title('Importancia de predictores (árbol)');

%%
% Entrena un árbol una vez (o usa el tuyo podado)
mdl = treeFinal;

baseAcc = mean(predict(mdl, T(:,featNames)) == Y);

drop = zeros(1,numel(featNames));
for j = 1:numel(featNames)
    Tperm = T(:,featNames);
    Tperm.(featNames{j}) = Tperm.(featNames{j})(randperm(height(Tperm)));
    accj = mean(predict(mdl, Tperm) == Y);
    drop(j) = baseAcc - accj;   % cuanto empeora al permutar
end

[dropS, idx] = sort(drop, 'descend');
disp(table(string(featNames(idx))', dropS', 'VariableNames', {'Feature','AccDrop'}));

%%
Ystr = string(Y);
for j = 1:numel(featNames)
    x = T.(featNames{j});
    x09 = x(Ystr=="09");
    x12 = x(Ystr=="12");

    p = ranksum(x09, x12); % no paramétrico, robusto
    fprintf('%-20s  p=%.3g  mean09=%.4f  mean12=%.4f\n', featNames{j}, p, mean(x09), mean(x12));
end