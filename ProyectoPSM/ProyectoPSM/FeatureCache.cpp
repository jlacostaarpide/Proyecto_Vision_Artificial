#include "FeatureCache.h"
#include <filesystem>
#include <regex>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;
using namespace cv;
using std::string;
using std::vector;

// =========================================================
// 1. FUNCIONES AUXILIARES (Estáticas para uso interno)
// =========================================================

static const vector<string> exts = { ".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp" };

// Comprueba si la extensión del archivo es válida
static bool hasSupportedExt(const fs::path& p) {
    string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return std::find(exts.begin(), exts.end(), e) != exts.end();
}

// Extrae la etiqueta (Clase) del nombre del archivo (ej: "02_pieza.jpg" -> 2)
static bool parseGTfromFilenameInt(const string& fname, int& outLabel) {
    // Busca 1 o 2 dígitos al principio del nombre
    std::regex rx(R"(^(\d{1,2}))");
    std::smatch m;
    if (std::regex_search(fname, m, rx) && m.size() >= 2) {
        try {
            outLabel = std::stoi(m[1].str());
            return true;
        }
        catch (...) { return false; }
    }
    return false;
}

// =========================================================
// 2. IMPLEMENTACIÓN DEL NAMESPACE FeatureCache
// =========================================================

namespace FeatureCache {

    static std::string kExtractorVersion() {
        return "v2_GENERIC_BATCH";
    }

    bool BuildFromFolder(const std::string& segFolder,
        ExtractorFunc extractor,
        FeatureCacheData& out,
        int* outSkippedNoGT, int* outSkippedBad)
    {
        // Resetear estructura de salida
        out = FeatureCacheData{};
        out.extractorVersion = kExtractorVersion();

        if (outSkippedNoGT) *outSkippedNoGT = 0;
        if (outSkippedBad)  *outSkippedBad = 0;

        if (!fs::exists(segFolder) || !fs::is_directory(segFolder)) return false;

        // 1. Listar archivos válidos
        vector<fs::path> files;
        for (auto& e : fs::directory_iterator(segFolder)) {
            if (e.is_regular_file() && hasSupportedExt(e.path())) {
                files.push_back(e.path());
            }
        }
        std::sort(files.begin(), files.end()); // Ordenar alfabéticamente

        if (files.empty()) return false;

        Mat X, y;
        bool namesSet = false;
        int expectedFeatures = 0;

        // 2. Recorrer archivos y procesar
        for (const auto& p : files) {
            string fname = p.filename().string();

            // A. Obtener Ground Truth del nombre
            int label = -1;
            if (!parseGTfromFilenameInt(fname, label)) {
                if (outSkippedNoGT) (*outSkippedNoGT)++;
                continue;
            }

            // B. Cargar Imagen
            // (Nota: Si tienes problemas con caracteres especiales aquí también, 
            // podrías necesitar el truco del imdecode, pero fs::path suele portarse bien).
            Mat I = imread(p.string(), IMREAD_COLOR);
            if (I.empty()) {
                if (outSkippedBad) (*outSkippedBad)++;
                continue;
            }

            // C. Ejecutar la función extractora (Callback)
            vector<double> feats;
            vector<string> names;

            bool ok = extractor(I, feats, names);

            if (!ok || feats.empty()) {
                if (outSkippedBad) (*outSkippedBad)++;
                continue;
            }

            // D. Comprobación de dimensiones (consistencia)
            if (!namesSet) {
                out.featNames = names;
                expectedFeatures = (int)feats.size();
                namesSet = true;
            }
            else {
                if (feats.size() != expectedFeatures) {
                    // Inconsistencia en el número de características
                    if (outSkippedBad) (*outSkippedBad)++;
                    continue;
                }
            }

            // E. Guardar en Matriz X
            Mat row(1, expectedFeatures, CV_32F);
            for (int k = 0; k < expectedFeatures; ++k) row.at<float>(0, k) = (float)feats[k];
            X.push_back(row);

            // F. Guardar Etiqueta y Nombre
            y.push_back(label);
            out.filenames.push_back(fname);
        }

        if (X.empty()) return false;

        // Convertir etiquetas a CV_32S (Enteros con signo)
        Mat yInt;
        Mat(y).convertTo(yInt, CV_32S);

        out.X = X;
        out.y = yInt;
        return true;
    }

    bool SaveYml(const std::string& ymlPath, const FeatureCacheData& d)
    {
        fs::path outp(ymlPath);
        // Crear directorios si no existen
        if (!outp.parent_path().empty()) fs::create_directories(outp.parent_path());

        cv::FileStorage fsw(ymlPath, cv::FileStorage::WRITE);
        if (!fsw.isOpened()) return false;

        fsw << "extractor_version" << d.extractorVersion;
        fsw << "X" << d.X;
        fsw << "y" << d.y;

        fsw << "filenames" << "[";
        for (auto& s : d.filenames) fsw << s;
        fsw << "]";

        fsw << "featNames" << "[";
        for (auto& s : d.featNames) fsw << s;
        fsw << "]";

        fsw.release();
        return true;
    }

    bool LoadYml(const std::string& ymlPath, FeatureCacheData& d)
    {
        cv::FileStorage fsr(ymlPath, cv::FileStorage::READ);
        if (!fsr.isOpened()) return false;

        d = FeatureCacheData{}; // Limpiar estructura

        fsr["extractor_version"] >> d.extractorVersion;
        fsr["X"] >> d.X;
        fsr["y"] >> d.y;

        // Leer secuencias (Arrays de strings)
        cv::FileNode fn = fsr["filenames"];
        if (!fn.empty()) {
            for (auto it = fn.begin(); it != fn.end(); ++it) {
                d.filenames.push_back((string)*it);
            }
        }

        cv::FileNode fnNames = fsr["featNames"];
        if (!fnNames.empty()) {
            for (auto it = fnNames.begin(); it != fnNames.end(); ++it) {
                d.featNames.push_back((string)*it);
            }
        }

        fsr.release();

        // Devolvemos true si al menos cargó la matriz de datos
        return !d.X.empty();
    }

} // namespace FeatureCache