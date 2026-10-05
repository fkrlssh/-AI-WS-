"""YOLO11 전이학습 (NEU-DET 6종)

GPU 환경에서 실행 (Colab / 경남TP GPU 서버). CPU로는 매우 느림.

  pip install ultralytics
  python scripts/prepare_data.py
  python vision/train.py --model yolo11n.pt --epochs 100 --imgsz 640
  python vision/train.py --data data/neu-det-aug/data.yaml --epochs 60 --name neu_aug_yolo11n   # 증강 데이터

※ yolo11n.pt 사전학습 가중치는 '기존자산'이므로 docs/SOURCES.md에 기록 (Ultralytics, AGPL-3.0)
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/neu-det/data.yaml")  # 증강 데이터: data/neu-det-aug/data.yaml
    ap.add_argument("--model", default="yolo11n.pt")      # n → s로 올리면 정확도↑ 속도↓
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)    # 원본 200x200, 업샘플로 작은 결함 검출 개선
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="neu_yolo11n")
    ap.add_argument("--device", default=None)             # 미지정 시 CUDA → Mac GPU(mps) → CPU 순 자동 선택
    a = ap.parse_args()

    import torch
    device = a.device or ("0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    model = YOLO(a.model)
    model.train(
        data=str(ROOT / a.data),
        epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=device,
        patience=25, seed=0, project=str(ROOT / "runs"), name=a.name,
        # 강판 표면: 상하좌우 뒤집기 OK, 색 변환은 약하게(회색조 이미지)
        fliplr=0.5, flipud=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, mosaic=1.0,
    )
    metrics = model.val(device=device)
    print(f"mAP@0.5 = {metrics.box.map50:.3f}   mAP@0.5:0.95 = {metrics.box.map:.3f}")
    for i, name in model.names.items():
        print(f"  {name:<16} AP50={metrics.box.ap50[i]:.3f}")
    print("best weights:", Path(model.trainer.best))
