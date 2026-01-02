#include "SegmentarGuardar.h"

#include "Segmentacion.h"  
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

// Extensiones permitidas
static bool hasSupportedExt(const fs::path& p)
{
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    static const std::vector<std::string> exts = {
        ".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"
    };
    return std::find(exts.begin(), exts.end(), e) != exts.end();
}

// Crea carpeta (y padres) si no existe
static bool ensureDir(const fs::path& dir)
{
    std::error_code ec;
    if (fs::exists(dir, ec)) return fs::is_directory(dir, ec);
    return fs::create_directories(dir, ec);
}

// Convierte a nombre base “segmented_<orig>_<k>.png”
static std::string buildCropName(const fs::path& imgPath, int k /*1-based*/)
{
    // ejemplo: 09_000_70_004.jpg -> segmented_09_000_70_004_a.png
    std::string stem = imgPath.stem().string();
    char suffix = (k >= 1 && k <= 26) ? char('a' + (k - 1)) : 'x';
    return stem + "_" + std::string(1, suffix) + ".png";
}

SegmentBatchStats SegmentFolderAndSaveCrops(
    const std::string& inputFolder,
    const std::string& outputFolder,
    bool keepSubfolders,
    int maxPiecesPerImage,
    bool verbose
)
{
    SegmentBatchStats st;

    fs::path inRoot(inputFolder);
    fs::path outRoot(outputFolder);

    std::error_code ec;
    if (!fs::exists(inRoot, ec) || !fs::is_directory(inRoot, ec)) {
        if (verbose) std::cerr << "[BatchSeg] Input folder no existe o no es carpeta: " << inputFolder << "\n";
        return st;
    }

    if (!ensureDir(outRoot)) {
        if (verbose) std::cerr << "[BatchSeg] No se pudo crear output folder: " << outputFolder << "\n";
        return st;
    }

    // Iteración recursiva si keepSubfolders, si no: no recursiva
    auto processOneImage = [&](const fs::path& imgPath)
        {
            st.images_total++;

            cv::Mat img = cv::imread(imgPath.string(), cv::IMREAD_COLOR);
            if (img.empty()) {
                st.images_failed_read++;
                if (verbose) std::cerr << "[BatchSeg] No se pudo leer: " << imgPath.string() << "\n";
                return;
            }
            st.images_read_ok++;

            // Segmentar (sin debug)
            std::vector<ResultadoPieza> piezas = Segmentacion::Segmentar(img, nullptr);

            if (piezas.empty()) {
                st.images_no_pieces++;
                if (verbose) std::cout << "[BatchSeg] 0 piezas: " << imgPath.filename().string() << "\n";
                return;
            }
            st.images_with_pieces++;

            // Carpeta destino
            fs::path outDir = outRoot;
            if (keepSubfolders) {
                // ruta relativa respecto a inRoot
                fs::path rel = fs::relative(imgPath.parent_path(), inRoot, ec);
                if (!ec && !rel.empty()) outDir /= rel;
            }
            if (!ensureDir(outDir)) {
                if (verbose) std::cerr << "[BatchSeg] No se pudo crear subcarpeta destino: " << outDir.string() << "\n";
                // contamos como fallos de guardado para todas las piezas
                st.crops_failed_save += (int)piezas.size();
                return;
            }

            int toSave = (maxPiecesPerImage > 0) ? std::min<int>((int)piezas.size(), maxPiecesPerImage)
                : (int)piezas.size();

            for (int i = 0; i < toSave; ++i) {
                const auto& res = piezas[i];
                if (res.imagenRecortada.empty()) {
                    st.crops_failed_save++;
                    continue;
                }

                fs::path outFile = outDir / buildCropName(imgPath, i + 1);

                // Guardar PNG (mejor para no degradar)
                bool ok = false;
                try {
                    ok = cv::imwrite(outFile.string(), res.imagenRecortada);
                }
                catch (...) {
                    ok = false;
                }

                if (ok) st.crops_saved++;
                else {
                    st.crops_failed_save++;
                    if (verbose) std::cerr << "[BatchSeg] Fallo guardando: " << outFile.string() << "\n";
                }
            }

            if (verbose) {
                std::cout << "[BatchSeg] " << imgPath.filename().string()
                    << " -> piezas: " << piezas.size()
                    << " | guardadas: " << toSave << "\n";
            }
        };

    if (keepSubfolders) {
        for (auto it = fs::recursive_directory_iterator(inRoot, ec);
            !ec && it != fs::recursive_directory_iterator(); ++it)
        {
            if (!it->is_regular_file()) continue;
            const fs::path p = it->path();
            if (!hasSupportedExt(p)) continue;
            processOneImage(p);
        }
    }
    else {
        for (auto& entry : fs::directory_iterator(inRoot, ec)) {
            if (!entry.is_regular_file()) continue;
            const fs::path p = entry.path();
            if (!hasSupportedExt(p)) continue;
            processOneImage(p);
        }
    }

    if (verbose) {
        std::cout << "\n=== BatchSeg summary ===\n";
        std::cout << "Total imágenes: " << st.images_total << "\n";
        std::cout << "Leídas OK:      " << st.images_read_ok << "\n";
        std::cout << "Fallos lectura: " << st.images_failed_read << "\n";
        std::cout << "Con piezas:     " << st.images_with_pieces << "\n";
        std::cout << "Sin piezas:     " << st.images_no_pieces << "\n";
        std::cout << "Crops guardados:" << st.crops_saved << "\n";
        std::cout << "Crops fallidos: " << st.crops_failed_save << "\n";
        std::cout << "========================\n";
    }

    return st;
}
