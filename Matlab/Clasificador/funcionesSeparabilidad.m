% funcionesSeparabilidad.m
% Requiere Xtrain (N x 8), Ytrain (categorical N x 1)

featureNames = {'HmeanCirc','HvarCirc', ...
                'Smedian','SIQR', ...
                'Vmedian','VIQR', ...
                'Smean','Vmean'};

%% 1) gplotmatrix
figure;
gplotmatrix(Xtrain, [], Ytrain, 'brg', '.', [], 'on', 'hist', ...
            featureNames, featureNames);
title('Matriz de dispersión de características de color (robustas)');

%% 2) Silhouette
[~,~,g] = unique(Ytrain);
figure;
silhouette(Xtrain, g);
xlabel('Silhouette value');
ylabel('Observación');
title('Índice de Silhouette por clase');
grid on;

%% Solo con HmeanCirc y Smean, por ejemplo
Xsub = Xtrain(:, [1 7]);    % columnas que quieras
[~,~,g] = unique(Ytrain);
figure;
silhouette(Xsub, g);
title('Silhouette solo con HmeanCirc y Smean');

%% 3) Histogramas de HmeanCirc por clase
HmeanCirc = Xtrain(:,1);
clases = categories(Ytrain);

figure; hold on;
for i = 1:numel(clases)
    idx = Ytrain == clases{i};
    histogram(HmeanCirc(idx), ...
              'DisplayName', char(clases{i}), ...
              'Normalization','probability', ...
              'FaceAlpha',0.5);
end
legend('Location','best');
xlabel('HmeanCirc');
ylabel('Probabilidad');
title('Distribución de HmeanCirc por clase');
grid on;
hold off;

%% 4) Boxplot de HmeanCirc por clase
for i=1:8
    Carac = Xtrain(:,i);
    figure;
    boxplot(Carac, Ytrain);
    xlabel('Clase');
    ylabel('HmeanCirc');
    title('Boxplot de HmeanCirc por clase');
end


%%

gplotmatrix(table2array(T_train(:,1:end-3)),[],L_train)

%%

figure; gplotmatrix(table2array(T(:,1:end-3)),[],T.Label); % como le hemos metido 3 columnas nuevas que no queremos ver, hacemos desde columna 1 hasta end-3.
figure; gplotmatrix(table2array(T2(:,1:end-3)),[],T2.Label);
figure; gplotmatrix(table2array(T3(:,1:end-3)),[],T3.Label);

%%
figure; gplotmatrix(table2array(T(:,1:4)),[],T.Label,[],'*'); % como le hemos metido 3 columnas nuevas que no queremos ver, hacemos desde columna 1 hasta end-3.
figure; gplotmatrix(table2array(T2(:,1:4)),[],T2.Label,[],'*');
figure; gplotmatrix(table2array(T3(:,1:4)),[],T3.Label,[],'*');


%%

% Train:
while 1, pause, n=randi(1200); fprintf('+ %d - %d\n',trainedModel.predictFcn(T_train(n,1:end-3)),L_train(n)), end

% Test
while 1, pause, n=randi(352); fprintf('+ %d - %d\n',trainedModel.predictFcn(T_test(n,1:end-3)),L_test(n)), end






