%% CLASIFICADOR LEGO 24 CLASES (código × orientación horizontal)
% Clase = pieza (código) + orientación horizontal (0..315 cada 45º)
% Orientación vertical NO define clase (variabilidad intra-clase)

clear; close all; clc;

%% ====== CONFIG ======
rootFolder = uigetdir(pwd, 'Selecciona carpeta raíz con las imágenes');
if rootFolder==0, error('No se seleccionó carpeta'); end

exts = {'*.jpg','*.jpeg','*.png'};
imgs = [];
for e = 1:numel(exts)
    imgs = [imgs; dir(fullfile(rootFolder, '**', exts{e}))];
end
if isempty(imgs), error('No se encontraron imágenes en la carpeta'); end

fprintf('Encontradas %d imágenes.\n', numel(imgs));

%% ====== 1) LEER NOMBRES Y CONSTRUIR MAPA DE CÓDIGOS ======
codes_all = zeros(numel(imgs),1);
oris_all  = zeros(numel(imgs),1);

for k = 1:numel(imgs)
    fname = imgs(k).name;
    tok = regexp(fname, '(\d+)_([0-9]+)_', 'tokens');
    if isempty(tok)
        error('Nombre no cumple patrón XX_YYY_... : %s', fname);
    end
    codes_all(k) = str2double(tok{1}{1});   % ej. 07 -> 7
    oris_all(k)  = str2double(tok{1}{2});   % ej. 270
end

codes_unique = unique(codes_all);
nCodes = numel(codes_unique);
fprintf('Códigos detectados: %s\n', mat2str(codes_unique));

if nCodes ~= 3
    warning('Se esperaban 3 códigos. Se detectaron %d. El script adapta clases igualmente.', nCodes);
end

cod_map = containers.Map(codes_unique, 1:nCodes);

%% ====== 2) CONSTRUIR DATASET X (features) y G (labels) ======
X = [];
G = [];

for k = 1:numel(imgs)
    fpath = fullfile(imgs(k).folder, imgs(k).name);
    I = imread(fpath);
    I = im2double(I);

    % --- segmentación ---
    mask = segment_lego(I);

    % --- features ---
    feat = extract_lego_features(I, mask);
    X = [X; feat];

    % --- etiqueta ---
    cod = codes_all(k);
    ori = oris_all(k);

    if mod(ori,45) ~= 0
        warning('Orientación no múltiplo de 45: %d en %s. Se redondea.', ori, imgs(k).name);
        ori = round(ori/45)*45;
    end
    ori_index = ori/45 + 1;   % 0->1, 45->2, ..., 315->8

    class = 8*(cod_map(cod)-1) + ori_index;  % 1..(8*nCodes)
    G = [G; class];
end

G = G(:);
fprintf('Dataset: X=%dx%d, G=%dx1, clases=%d\n', size(X,1), size(X,2), numel(G), numel(unique(G)));

%% ====== 3) NORMALIZACIÓN ROBUSTA (p5-p95) ======
p5  = prctile(X,5);
p95 = prctile(X,95);
Xn = (X - p5) ./ (p95 - p5 + eps);

%% ====== 4) ENTRENAR CLASIFICADORES (5-fold CV) ======
cv = cvpartition(G,'KFold',5);

% LDA
lda = fitcdiscr(Xn, G, 'DiscrimType','linear', 'CVPartition',cv);
acc_lda = 1 - kfoldLoss(lda);

% kNN
knn = fitcknn(Xn, G, 'NumNeighbors',5, 'Standardize',false, 'CVPartition',cv);
acc_knn = 1 - kfoldLoss(knn);

% SVM ECOC (RBF)
tSVM = templateSVM('KernelFunction','rbf','KernelScale','auto','Standardize',false);
svm = fitcecoc(Xn, G, 'Learners',tSVM, 'Coding','onevsone', 'CVPartition',cv);
acc_svm = 1 - kfoldLoss(svm);

% Árbol
tree = fitctree(Xn, G, 'CVPartition',cv);
acc_tree = 1 - kfoldLoss(tree);

fprintf('\nACCURACY 5-fold:\n');
fprintf('LDA   : %.3f\n', acc_lda);
fprintf('kNN   : %.3f\n', acc_knn);
fprintf('SVM   : %.3f\n', acc_svm);
fprintf('TREE  : %.3f\n', acc_tree);

[bestAcc, bestIdx] = max([acc_lda, acc_knn, acc_svm, acc_tree]);
bestNames = {'LDA','kNN','SVM-ECOC','TREE'};
fprintf('\nMejor clasificador: %s (%.3f)\n', bestNames{bestIdx}, bestAcc);

%% ====== 5) MATRIZ DE CONFUSIÓN DEL MEJOR ======
% Reentrenar el mejor SIN CV para predecir todo (o usa pFoldPredict si prefieres).
switch bestIdx
    case 1
        model = fitcdiscr(Xn,G,'DiscrimType','linear');
    case 2
        model = fitcknn(Xn,G,'NumNeighbors',5,'Standardize',false);
    case 3
        model = fitcecoc(Xn,G,'Learners',tSVM,'Coding','onevsone');
    case 4
        model = fitctree(Xn,G);
end

Gpred = predict(model, Xn);
C = confusionmat(G, Gpred);

figure('Name','Matriz de Confusión');
heatmap(C);
title(sprintf('Confusion Matrix (%s)', bestNames{bestIdx}));

TA_total = sum(diag(C))/sum(C(:));
fprintf('\nTA total (train-set): %.3f\n', TA_total);

%% ====== 6) MÉTRICAS POR CLASE ======
nClasses = size(C,1);
TPR = zeros(nClasses,1);
TNR = zeros(nClasses,1);

for c = 1:nClasses
    TP = C(c,c);
    FN = sum(C(c,:)) - TP;
    FP = sum(C(:,c)) - TP;
    TN = sum(C(:)) - (TP+FN+FP);

    TPR(c) = TP/(TP+FN+eps);
    TNR(c) = TN/(TN+FP+eps);
end

figure('Name','TPR (Sensibilidad) por clase');
stem(TPR); ylim([0 1]); grid on;
title('TPR por clase');

figure('Name','TNR (Especificidad) por clase');
stem(TNR); ylim([0 1]); grid on;
title('TNR por clase');

%% ========================================================================
%% ======================== FUNCIONES LOCALES =============================
%% ========================================================================

function mask_final = segment_lego(I)
% Segmentación robusta y rápida:
% - L* (Lab)
% - top-hat para iluminación
% - umbral local adaptativo
% - morfología
% - watershed marcado SOLO en ROI y a menor resolución

I = im2double(I);

% Luminancia L*
lab = rgb2lab(I);
Lch = mat2gray(lab(:,:,1));
area_img = numel(Lch);

% Top-hat (fondo por apertura grande)
seR = round(sqrt(area_img)/120);
seR = max(seR, 35);
bg = imopen(Lch, strel('disk', seR));
In = mat2gray(Lch - bg);

% Suavizado fuerte
In_s = imgaussfilt(In, 2.0);
In_s = medfilt2(In_s, [5 5]);

% Gamma adaptativa
V_med = median(In_s(:));
gamma = 1.0 + 0.6*(0.5 - V_med);
gamma = max(min(gamma,1.4),0.7);
In_g = In_s.^gamma;

% Bordes Canny
E = edge(In_g, 'canny', [], 1.8);
E = imdilate(E, strel('disk',1));

% Umbral local adaptativo (menos sensibilidad para no salpicar)
win = round(min(size(In_g))/4);
win = max(win, 71);
if mod(win,2)==0, win=win+1; end
T = adaptthresh(In_g, 0.28, 'NeighborhoodSize', win);
mask_int = imbinarize(In_g, T);

% Limpieza previa fuerte
mask_int = bwareaopen(mask_int, round(0.0006*area_img));
mask_int = imclose(mask_int, strel('disk',3));
mask_int = imfill(mask_int,'holes');

% Combinar bordes cerca de objetos
near_obj = imdilate(mask_int, strel('disk',4));
mask0 = mask_int | (E & near_obj);

% Morfología
mask = bwmorph(mask0,'clean');
mask = bwmorph(mask,'majority');
mask = imopen(mask, strel('disk',2));
mask = imclose(mask, strel('disk',3));
mask = imfill(mask,'holes');
mask = bwareaopen(mask, max(400, round(0.00008*area_img)));

% Si no hay nada, salir
if ~any(mask(:))
    mask_final = mask;
    return;
end

% ========= ROI para acelerar =========
props = regionprops(mask,'BoundingBox','Area');
[~,idMax] = max([props.Area]);
bb = props(idMax).BoundingBox;

pad = 20; % margen
x1 = max(floor(bb(1))-pad,1);
y1 = max(floor(bb(2))-pad,1);
x2 = min(ceil(bb(1)+bb(3))+pad, size(mask,2));
y2 = min(ceil(bb(2)+bb(4))+pad, size(mask,1));

mask_roi = mask(y1:y2, x1:x2);

% ========= Watershed marcado en ROI y downsample =========
scale = 0.5;  % 0.5 = mitad de resolución solo para WS
mask_small = imresize(mask_roi, scale, 'nearest');

D = bwdist(~mask_small);
D = imgaussfilt(D, 1);

% Marcadores: evitar sobresegmentación
% extendedmin con valor adaptado al tamaño
minimos = imextendedmin(D, 0.25);   % 0.2–0.35
D2 = imimposemin(-D, minimos);

Lw = watershed(D2);
mask_ws = mask_small;
mask_ws(Lw==0) = 0;

% Reescalar WS a ROI original
mask_ws = imresize(mask_ws, size(mask_roi), 'nearest');

% Sustituir ROI en la máscara global
mask2 = mask;
mask2(y1:y2, x1:x2) = mask_ws;

% Quitar borde + filtro geométrico final
mask2 = imclearborder(mask2);

[Llbl, ~] = bwlabel(mask2, 8);
stats = regionprops(Llbl,'Area','Extent','Eccentricity');

if isempty(stats)
    mask_final = mask2;
    return;
end

areas = [stats.Area];
ext   = [stats.Extent];
ecc   = [stats.Eccentricity];

minA = max(400, round(0.00008*area_img));
maxA = 0.20*area_img;

idx_ok = find(areas>minA & areas<maxA & ext>0.15 & ecc<0.999);
mask_final = ismember(Llbl, idx_ok);
mask_final = logical(mask_final);

end

function feat = extract_lego_features(I, mask)
% Features mixtas:
% Color (Lab), forma (región), orientación (PCA/regionprops),
% + Hu moments (rotación-invar y dependientes).

I = im2double(I);
mask = logical(mask);
if ~any(mask(:))
    feat = zeros(1,25);
    return;
end

% quedarse con el objeto principal (mayor área)
CC = bwconncomp(mask);
numPix = cellfun(@numel, CC.PixelIdxList);
[~,idMax] = max(numPix);
maskMain = false(size(mask));
maskMain(CC.PixelIdxList{idMax}) = true;

% regionprops geométricos
stats = regionprops(maskMain, 'Area','Perimeter','BoundingBox','Centroid',...
    'Eccentricity','Extent','Solidity','Orientation','MajorAxisLength','MinorAxisLength');

A  = stats.Area;
P  = stats.Perimeter;
BB = stats.BoundingBox;
w  = BB(3); h = BB(4);
aspect = w/(h+eps);
circ = 4*pi*A/(P^2+eps);

ecc = stats.Eccentricity;
ext = stats.Extent;
sol = stats.Solidity;
ori = stats.Orientation; % grados (rotación sensible)

maj = stats.MajorAxisLength;
minr = stats.MinorAxisLength;

% color en Lab dentro del objeto
lab = rgb2lab(I);
L = lab(:,:,1); a = lab(:,:,2); b = lab(:,:,3);
L_m = mean(L(maskMain));  L_s = std(L(maskMain));
a_m = mean(a(maskMain));  a_s = std(a(maskMain));
b_m = mean(b(maskMain));  b_s = std(b(maskMain));
C_m = mean( sqrt(a(maskMain).^2 + b(maskMain).^2) );

% Hu moments sobre máscara
hu = humoments(maskMain);

% proyecciones (rotación sensible)
projX = sum(maskMain,1);
projY = sum(maskMain,2);
projX = projX / (max(projX)+eps);
projY = projY / (max(projY)+eps);

% 4 momentos simples de proyecciones
px_mean = mean(projX); px_std = std(projX);
py_mean = mean(projY); py_std = std(projY);

feat = [ ...
    A, P, circ, aspect, ecc, ext, sol, ori, maj, minr, ...
    L_m, L_s, a_m, a_s, b_m, b_s, C_m, ...
    hu(:).', ...
    px_mean, px_std, py_mean, py_std ...
    ];
end

function hu = humoments(BW)
% Hu moments (7)
BW = logical(BW);
BW = bwfill(BW,'holes');
BW = bwareaopen(BW, 20);

[m,n] = size(BW);
[x,y] = meshgrid(1:n,1:m);

M00 = sum(BW(:));
xbar = sum(x(:).*BW(:))/M00;
ybar = sum(y(:).*BW(:))/M00;

x2 = x - xbar;
y2 = y - ybar;

mu11 = sum(sum((x2.*y2).*BW));
mu20 = sum(sum((x2.^2).*BW));
mu02 = sum(sum((y2.^2).*BW));
mu30 = sum(sum((x2.^3).*BW));
mu03 = sum(sum((y2.^3).*BW));
mu21 = sum(sum((x2.^2.*y2).*BW));
mu12 = sum(sum((x2.*y2.^2).*BW));

% normalizados
eta11 = mu11/(M00^(1+ (1+1)/2));
eta20 = mu20/(M00^(1+ (2+0)/2));
eta02 = mu02/(M00^(1+ (0+2)/2));
eta30 = mu30/(M00^(1+ (3+0)/2));
eta03 = mu03/(M00^(1+ (0+3)/2));
eta21 = mu21/(M00^(1+ (2+1)/2));
eta12 = mu12/(M00^(1+ (1+2)/2));

hu = zeros(1,7);
hu(1) = eta20 + eta02;
hu(2) = (eta20 - eta02)^2 + 4*eta11^2;
hu(3) = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2;
hu(4) = (eta30 + eta12)^2 + (eta21 + eta03)^2;
hu(5) = (eta30 - 3*eta12)*(eta30+eta12)*((eta30+eta12)^2 - 3*(eta21+eta03)^2) + ...
        (3*eta21 - eta03)*(eta21+eta03)*(3*(eta30+eta12)^2 - (eta21+eta03)^2);
hu(6) = (eta20 - eta02)*((eta30+eta12)^2 - (eta21+eta03)^2) + ...
        4*eta11*(eta30+eta12)*(eta21+eta03);
hu(7) = (3*eta21 - eta03)*(eta30+eta12)*((eta30+eta12)^2 - 3*(eta21+eta03)^2) - ...
        (eta30 - 3*eta12)*(eta21+eta03)*(3*(eta30+eta12)^2 - (eta21+eta03)^2);
end
