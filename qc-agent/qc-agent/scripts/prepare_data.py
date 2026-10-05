"""NEU-DET(강판 표면결함, 6종, YOLO 라벨) 다운로드 및 정리.

원본: Northeastern University(NEU) Surface Defect Database (Song Kechen et al.)
미러: github.com/Marfbin/NEU-DET-with-yolov8 (YOLO 형식 변환본)  → docs/SOURCES.md에 기록

결과:
  data/neu-det/{train,val}/{images,labels}
  data/neu-det/data.yaml
"""
import io, shutil, zipfile, urllib.request
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "neu-det"
URL = "https://codeload.github.com/Marfbin/NEU-DET-with-yolov8/zip/refs/heads/main"
CLASSES = yaml.safe_load(open(ROOT / "config/process_spec.yaml", encoding="utf-8"))["classes"]


def main(zip_path: str | None = None):
    if (OUT / "data.yaml").exists():
        print(f"[skip] 이미 존재: {OUT}")
        return
    if zip_path:
        zf = zipfile.ZipFile(zip_path)
    else:
        print("[download]", URL)
        zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))

    prefix = "NEU-DET-with-yolov8-main/data/NEU-DET/"
    split_map = {"train": "train", "test": "val"}  # 원본 test → val로 사용
    n = 0
    for name in zf.namelist():
        if not name.startswith(prefix) or name.endswith("/"):
            continue
        parts = name[len(prefix):].split("/")   # train/images/xxx.jpg
        if len(parts) != 3 or parts[0] not in split_map or parts[1] not in ("images", "labels"):
            continue                              # *.cache 등 불필요 파일 제외
        split, kind, fname = parts
        dst = OUT / split_map[split] / kind / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src, open(dst, "wb") as f:
            shutil.copyfileobj(src, f)
        n += 1

    data_yaml = {"path": str(OUT), "train": "train/images", "val": "val/images",
                 "nc": len(CLASSES), "names": CLASSES}
    yaml.safe_dump(data_yaml, open(OUT / "data.yaml", "w"), allow_unicode=True, sort_keys=False)
    for s in ("train", "val"):
        print(f"  {s}: {len(list((OUT/s/'images').glob('*.jpg')))} images")
    print(f"[done] {n} files → {OUT}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
