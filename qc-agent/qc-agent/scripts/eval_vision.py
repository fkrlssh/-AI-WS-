"""비전 성능 평가 — 제안서 목표치(분류 ≥ 90%, 1장당 ≤ 100ms, 정상 오검출) 측정

  python scripts/eval_vision.py --weights runs/neu_yolo11n/weights/best.pt --normals "../../경남 AISW/pn_data/pseudo_normal/test"

측정 항목
1) 이미지 단위 분류 정확도: NEU-DET val 이미지마다 최고 신뢰도 박스의 클래스 = 정답 클래스(파일명 접두어)이면 정답.
   박스가 하나도 없으면 오답(미검출). 클래스별 정확도도 출력.
2) 추론 시간: 워밍업 5장 후 1장당 predict() 평균·p95 (ms). 장비 이름을 함께 기록할 것.
3) 정상 오검출률: 의사 정상 이미지 중 박스가 1개 이상 나온 비율 (학습에 쓰지 않은 test 폴더 사용).
결과는 runs/eval_vision.json 에 저장.
※ mAP@0.5는 학습 로그(runs/neu_yolo11n/results.csv)의 val 값을 쓴다.
"""
import argparse, json, platform, statistics, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def gt_of(name: str) -> str:
    for c in sorted(CLASSES, key=len, reverse=True):
        if name.startswith(c):
            return c
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/neu_bg_yolo11n/weights/best.pt")
    ap.add_argument("--val", default="data/neu-det/val/images")
    ap.add_argument("--normals", default="data/pseudo_normal", help="의사 정상 이미지 폴더(가능하면 test 분할)")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(ROOT / a.weights))
    names = model.names

    vals = sorted(p for p in (ROOT / a.val).glob("*") if p.suffix.lower() in (".jpg", ".png", ".bmp"))
    for p in vals[:5]:  # 워밍업
        model.predict(str(p), conf=a.conf, verbose=False)

    hit, tot, times = defaultdict(int), defaultdict(int), []
    for p in vals:
        t0 = time.perf_counter()
        r = model.predict(str(p), conf=a.conf, verbose=False)[0]
        times.append((time.perf_counter() - t0) * 1000)
        g = gt_of(p.name)
        tot[g] += 1
        if len(r.boxes):
            pred = names[int(r.boxes.cls[int(r.boxes.conf.argmax())])]
            hit[g] += pred == g

    n = sum(tot.values())
    acc = sum(hit.values()) / n
    print(f"[분류] val {n}장 이미지 단위 정확도 = {acc:.1%} (목표 ≥ 90%)")
    per = {c: round(hit[c] / tot[c], 3) for c in CLASSES if tot[c]}
    for c, v in per.items():
        print(f"   {c:16s} {v:.1%} ({hit[c]}/{tot[c]})")
    times.sort()
    mean_ms, p95_ms = statistics.mean(times), times[int(len(times) * 0.95) - 1]
    print(f"[속도] 1장 평균 {mean_ms:.1f} ms, p95 {p95_ms:.1f} ms (목표 ≤ 100ms) · 장비 {platform.platform()} / {platform.processor()}")

    normals = sorted((ROOT / a.normals).rglob("*.png"))
    fp = sum(1 for p in normals if len(model.predict(str(p), conf=a.conf, verbose=False)[0].boxes))
    fpr = fp / len(normals) if normals else None
    if normals:
        print(f"[오검출] 의사 정상 {len(normals)}장 중 {fp}장에서 결함 검출 → 오검출률 {fpr:.1%}")

    out = {"weights": a.weights, "conf": a.conf, "val_images": n, "image_accuracy": round(acc, 4), "per_class": per,
           "latency_ms_mean": round(mean_ms, 1), "latency_ms_p95": round(p95_ms, 1), "device": platform.platform(),
           "normals": len(normals), "false_positive_rate": None if fpr is None else round(fpr, 4)}
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/eval_vision.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved] runs/eval_vision.json")


if __name__ == "__main__":
    main()
