% funcionesSeparabilidad.m

%%
figure; gplotmatrix(table2array(F(:,1:4)),[],F.Label,[],'*'); % como le hemos metido 3 columnas nuevas que no queremos ver, hacemos desde columna 1 hasta end-3.

%%
% Test
while 1, pause, n=randi(size(F_test,1)); fprintf('+ %d - %d\n',trainedModel.predictFcn(F_test(n,1:end-3)),F_test.Label(n,1)), end

% 1 - 03
% 2 - 06
% 3 - 09
% 4 - 12



