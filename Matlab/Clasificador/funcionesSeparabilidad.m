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

