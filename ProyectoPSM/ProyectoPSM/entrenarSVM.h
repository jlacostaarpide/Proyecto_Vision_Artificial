#pragma once

#include <string>

namespace TrainSVM {

    // Opciones para el entrenamiento
    struct Options {
        std::string inputFolder;   // carpeta con imágenes segmentadas
        std::string outModelPath;  // ruta donde guardar modelo (.yml)
        std::string csvOut;        // ruta CSV para features
        double C = 1.0;            // parámetro C del SVM
        double gamma = 0.0;        // gamma (0 -> auto 1/n_features)
        bool doScale = false;      // aplicar z-score y guardar scaler
    };

    // Entrena un SVM polinómico de grado 2 (cuadrático).
    // Devuelve 0 si todo OK, distinto de 0 en caso de error.
    int TrainQuadraticSVM(const Options& opts);

} // namespace TrainSVM