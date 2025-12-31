function [label, score, featRow] = predict_9v12_fromImage(I, model, opts)
% I: imagen segmentada (fondo negro) RGB/gray
% model: struct cargado de tree_9v12_final.mat
% opts: opcional, se pasa a extractShapeFeatures

    if nargin < 3, opts = struct(); end

    % Tu extractor reducido (6 feats)
    [feat6, featNames6] = extractShapeFeatures(I, opts);

    % Asegurar orden idéntico al modelo
    [~, idx] = ismember(model.featNames, featNames6);
    if any(idx==0)
        error("Faltan features en extractShapeFeatures. Esperadas: %s", strjoin(model.featNames,", "));
    end
    featRow = feat6(idx);

    X = array2table(featRow, "VariableNames", model.featNames);

    [label, score] = predict(model.tree, X); % label categorical
    label = string(label);                  % "09" o "12"
end
