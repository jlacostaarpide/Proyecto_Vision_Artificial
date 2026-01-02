#pragma once
#include <string>

struct SegmentBatchStats
{
    int images_total = 0;
    int images_read_ok = 0;
    int images_failed_read = 0;

    int images_with_pieces = 0;
    int images_no_pieces = 0;

    int crops_saved = 0;
    int crops_failed_save = 0;
};

// Recorre inputFolder, segmenta cada imagen, y guarda los crops en outputFolder.
// - keepSubfolders: si true, replica estructura de subcarpetas en output.
// - maxPiecesPerImage: 0 = sin límite; si >0 guarda como máximo ese número por imagen (ya vienen ordenadas por área).
SegmentBatchStats SegmentFolderAndSaveCrops(
    const std::string& inputFolder,
    const std::string& outputFolder,
    bool keepSubfolders = false,
    int maxPiecesPerImage = 0,
    bool verbose = true
);
#pragma once
