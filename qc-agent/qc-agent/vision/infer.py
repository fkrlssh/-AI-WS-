"""검사 이미지 → 결함 판별 결과를 factory.db의 inspections 테이블에 기록

  python vision/infer.py --weights runs/neu_yolo11n/weights/best.pt   # 실제 YOLO 추론
  python vision/infer.py --mock                                       # 모델 학습 전 파이프라인 개발용

--mock 은 시뮬레이션 정답(gt)을 그대로 복사합니다(source='mock').
Agent/원인추적 개발을 모델 학습과 병렬로 진행하기 위한 임시 모드이며, 시연·성능 평가에는 사용 금지.
"""
import argparse, json, random, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def label_boxes(img_path: str):
    lbl = ROOT / img_path.replace("/images/", "/labels/").rsplit(".", 1)[0]
    lbl = Path(str(lbl) + ".txt")
    if not lbl.exists():
        return []
    return [[float(x) for x in line.split()[1:]] for line in lbl.read_text().splitlines() if line.strip()]


def run(weights=None, mock=False, conf_th=0.25, limit=None):
    db = sqlite3.connect(ROOT / "data/factory.db")
    rows = db.execute("SELECT product_id, image_path, gt_defect FROM inspections ORDER BY ts").fetchall()
    if limit:
        rows = rows[:limit]
    model = None
    if not mock:
        from ultralytics import YOLO
        model = YOLO(weights)

    out = []
    for pid, img, gt in rows:
        if img is None:  # NEU-DET에는 정상 이미지가 없음 → 정상 제품은 이미지 없이 통과 처리(추후 보완)
            out.append(("normal", None, None, "no_image", pid)); continue
        if mock:
            out.append((gt, round(random.uniform(0.8, 0.99), 3), json.dumps(label_boxes(img)), "mock", pid)); continue
        r = model.predict(str(ROOT / img), conf=conf_th, verbose=False)[0]
        if len(r.boxes) == 0:
            out.append(("normal", None, None, "yolo", pid)); continue
        best = int(r.boxes.conf.argmax())
        cls = model.names[int(r.boxes.cls[best])]
        boxes = [[round(v, 4) for v in b] for b in r.boxes.xywhn.tolist()]
        out.append((cls, round(float(r.boxes.conf[best]), 3), json.dumps(boxes), "yolo", pid))

    db.executemany("UPDATE inspections SET pred_defect=?, conf=?, bbox_json=?, source=? WHERE product_id=?", out)
    db.commit()
    n_img = sum(1 for o in out if o[3] != "no_image")
    if n_img and not mock:
        correct = sum(1 for (pid, img, gt), o in zip(rows, out) if img and o[0] == gt)
        print(f"[yolo] 이미지 {n_img}장 분류 정확도(top-1, 시뮬레이션 스트림 기준) = {correct / n_img:.1%}")
    print(f"[done] {len(out)}건 기록 (mode={'mock' if mock else 'yolo'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    if not a.mock and not a.weights:
        ap.error("--weights 또는 --mock 중 하나 필요")
    run(a.weights, a.mock, a.conf)
