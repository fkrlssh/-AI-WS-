"""
YOLO11 결함 검출 학습 (로컬 PC)
  python train_yolo.py              # 증강 데이터(neu-det-aug)로 학습
  python train_yolo.py --base       # 원본(neu-det)으로도 학습해서 비교
  python train_yolo.py --quick      # 동작 확인용(3에폭)

결과: runs/<이름>/weights/best.pt, runs/compare.csv(검증 mAP 비교)
"""
import argparse, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def write_yaml(ds: Path) -> Path:
    """data.yaml의 path를 이 PC 경로로 맞춘 사본 생성"""
    y = ds / 'data_local.yaml'
    y.write_text('\n'.join([f'path: {ds.as_posix()}', 'train: train/images', 'val: val/images',
                            f'nc: {len(NAMES)}', 'names:'] + [f'- {n}' for n in NAMES]) + '\n', encoding='utf-8')
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', action='store_true', help='원본 데이터 학습도 함께 수행')
    ap.add_argument('--model', default='yolo11s.pt')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=-1, help='-1 = GPU 메모리에 맞춰 자동')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else 'cpu'
    print('device:', torch.cuda.get_device_name(0) if device == 0 else 'CPU (GPU가 없으면 매우 느림)')
    batch = a.batch if device == 0 else 16
    epochs = 3 if a.quick else a.epochs

    jobs = [('aug', ROOT / 'neu-det-aug')]
    if a.base:
        jobs.insert(0, ('base', ROOT / 'neu-det'))

    rows = []
    for name, ds in jobs:
        model = YOLO(a.model)
        model.train(
            data=str(write_yaml(ds)), epochs=epochs, imgsz=a.imgsz, batch=batch, device=device,
            project=str(ROOT / 'runs'), name=name, exist_ok=True, workers=4, seed=0, patience=20,
            # 흑백 강판 영상에 맞춘 온라인 증강
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=0.0, flipud=0.5, fliplr=0.5,
            mosaic=1.0, mixup=0.05, close_mosaic=10,
        )
        best = YOLO(str(ROOT / 'runs' / name / 'weights' / 'best.pt'))
        m = best.val(data=str(write_yaml(ROOT / 'neu-det')), imgsz=a.imgsz, device=device,
                     project=str(ROOT / 'runs'), name=f'{name}_val', exist_ok=True)
        row = {'run': name, 'mAP50': round(m.box.map50, 4), 'mAP50-95': round(m.box.map, 4)}
        for i, n in enumerate(NAMES):
            row[f'AP50_{n}'] = round(m.box.ap50[i], 4)
        rows.append(row)
        print(row)

    out = ROOT / 'runs' / 'compare.csv'
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('비교 결과:', out)


if __name__ == '__main__':   # Windows 멀티프로세싱 필수
    main()
