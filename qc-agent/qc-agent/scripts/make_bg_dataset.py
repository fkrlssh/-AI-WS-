"""배경(정상) 이미지를 넣은 YOLO 학습셋 만들기 — 정상 강판 오검출 줄이기

기존 모델은 결함 이미지만으로 학습해 '정상'을 본 적이 없다. 그래서 의사 정상 이미지의 67%를 불량으로 판정했다.
Ultralytics 권장 방식대로 정상 이미지를 '빈 라벨(.txt)' 배경 이미지로 학습셋에 넣는다.

- train = NEU-DET train 전체 + 의사 정상 이미지(라벨 없음)
- val   = NEU-DET val 그대로 (기존 모델과 mAP를 같은 기준으로 비교)
- val 원본(…_271~300)에서 만든 의사 정상 이미지는 학습에서 뺀다 → 오검출률 측정용으로 남김 (누수 방지)

  python scripts/make_bg_dataset.py --normals data/pseudo_normal
  python vision/train.py --data data/neu-det-bg/data.yaml --name neu_bg_yolo11n
"""
import argparse, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/neu-det")
ap.add_argument("--normals", default="data/pseudo_normal", help="의사 정상 png 폴더(하위 폴더 포함 검색)")
ap.add_argument("--out", default="data/neu-det-bg")
a = ap.parse_args()

src, out = ROOT / a.src, ROOT / a.out
if out.exists():
    shutil.rmtree(out)
for split in ("train", "val"):
    for sub in ("images", "labels"):
        shutil.copytree(src / split / sub, out / split / sub, ignore=shutil.ignore_patterns("*.cache"))

val_stems = {p.stem for p in (src / "val/images").iterdir()}
added = held = 0
for p in sorted((ROOT / a.normals).rglob("*.png")):
    source = p.stem.removesuffix("_pn")
    if source in val_stems:
        held += 1
        continue
    shutil.copy(p, out / "train/images" / p.name)
    (out / "train/labels" / f"{p.stem}.txt").write_text("")
    added += 1

names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
(out / "data.yaml").write_text(
    # path를 비워 두면 Ultralytics가 data.yaml이 있는 폴더를 기준으로 삼는다 → 폴더째 Mac으로 옮겨도 동작
    f"train: train/images\nval: val/images\nnc: 6\nnames: {names}\n", encoding="utf-8")
n_train = len(list((out / "train/images").iterdir()))
print(f"[done] {out} — train {n_train}장 (배경 {added}장, {added / n_train:.0%}), val 원본에서 만든 의사 정상 {held}장은 평가용으로 제외")
