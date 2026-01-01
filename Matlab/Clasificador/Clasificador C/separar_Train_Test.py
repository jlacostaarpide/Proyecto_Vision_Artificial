import os
import re
import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def get_label_from_name(fname: str):
    # tu código actual usa ^(\d{1,2}) : lo replico
    m = re.search(r"^(\d{1,2})", fname)
    return int(m.group(1)) if m else None

def list_images(folder: Path):
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS:
            files.append(p)
    files.sort()
    return files

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main(
    in_dir: str,
    out_train: str,
    out_test: str,
    test_ratio: float = 0.2,
    seed: int = 1234,
    min_test_per_class: int = 1,
    copy: bool = True
):
    in_dir = Path(in_dir)
    out_train = Path(out_train)
    out_test = Path(out_test)

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"ERROR: No existe carpeta: {in_dir}")

    safe_mkdir(out_train)
    safe_mkdir(out_test)

    files = list_images(in_dir)
    if not files:
        raise SystemExit(f"ERROR: No hay imágenes en {in_dir}")

    by_class = defaultdict(list)
    skipped = 0
    for p in files:
        lbl = get_label_from_name(p.name)
        if lbl is None:
            skipped += 1
            continue
        by_class[lbl].append(p)

    if not by_class:
        raise SystemExit("ERROR: No se pudo extraer ninguna clase (revisa el patrón de nombres).")

    rng = random.Random(seed)

    train_list = []
    test_list = []

    # split estratificado por clase
    for lbl, lst in sorted(by_class.items()):
        lst = lst.copy()
        rng.shuffle(lst)
        n = len(lst)

        # tamaño de test por clase
        n_test = int(round(n * test_ratio))
        n_test = max(min_test_per_class, n_test) if n > 1 else 0
        # evitar vaciar train o test si la clase es pequeña
        if n_test >= n:
            n_test = n - 1
        if n_test < 0:
            n_test = 0

        test_cls = lst[:n_test]
        train_cls = lst[n_test:]

        test_list.extend([(p, lbl) for p in test_cls])
        train_list.extend([(p, lbl) for p in train_cls])

    # copia/mueve
    def transfer(src: Path, dst_dir: Path):
        dst = dst_dir / src.name
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)

    # (opcional) limpiar destino si ya tenía cosas: aquí NO lo hago para no borrar sin querer
    for p, lbl in train_list:
        transfer(p, out_train)
    for p, lbl in test_list:
        transfer(p, out_test)

    # manifest para reproducibilidad
    manifest = (out_train.parent / "split_manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "split"])
        for p, lbl in train_list:
            w.writerow([p.name, lbl, "train"])
        for p, lbl in test_list:
            w.writerow([p.name, lbl, "test"])

    # resumen
    train_count = defaultdict(int)
    test_count = defaultdict(int)
    for _, lbl in train_list:
        train_count[lbl] += 1
    for _, lbl in test_list:
        test_count[lbl] += 1

    print("=== SPLIT DONE ===")
    print(f"Input: {in_dir}")
    print(f"Train: {out_train} ({len(train_list)} imgs)")
    print(f"Test : {out_test} ({len(test_list)} imgs)")
    print(f"Skipped (no label): {skipped}")
    print(f"Manifest: {manifest}")
    print("\nPer-class counts (train/test):")
    for lbl in sorted(by_class.keys()):
        print(f"  {lbl:02d}: {train_count[lbl]} / {test_count[lbl]} (total {train_count[lbl]+test_count[lbl]})")

if __name__ == "__main__":
    # AJUSTA AQUÍ TUS RUTAS
    main(
        in_dir=r"C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_AMARILLAS",
        out_train=r"C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_TRAIN_AMARILLAS",
        out_test=r"C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_TEST_AMARILLAS",
        test_ratio=0.15,
        seed=1234,
        min_test_per_class=1,
        copy=True
    )
