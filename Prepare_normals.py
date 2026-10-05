"""
NEU-DET 정상(무결함) 데이터 생성 스크립트
==========================================
NEU-DET는 6개 결함 클래스 x 300장이고 모든 이미지에 결함 박스가 있어 정상 이미지가 없다.
이 스크립트는 NEU-DET 안에서 정상 데이터를 두 가지 형태로 만든다.

  1) pseudo_normal/  : 결함 박스를 같은 이미지의 깨끗한 영역 텍스처로 대체(Poisson seamless cloning)한
                       200x200 '의사 정상' 이미지 → PatchCore 학습, 시연용 양품 흐름, YOLO 배경 이미지, 오검출률 측정
  2) normal_patches/ : 결함 박스 밖에서 잘라낸 원본 픽셀 그대로의 64x64 정상 패치 → 패치 단위 이상탐지용

정상 소스는 inclusion, scratches(패치는 pitted_surface 포함)만 쓴다.
crazing, patches, rolled-in_scale 은 박스 밖에도 결함 텍스처가 퍼져 있어 정상으로 쓰면 안 된다.

사용법
  python prepare_normals.py --neu-root NEU-DET-Steel-Surface-Defect-Detection --out neu_normals
  # YOLO 학습/평가 분할을 이미 정했다면, 평가(val/test) 이미지 이름 목록(한 줄에 하나, 확장자 무관)을 넘긴다
  python prepare_normals.py --neu-root ... --out ... --test-list my_val_images.txt

산출물
  out/pseudo_normal/{train,test}/*.png   (test = 평가용 원본에서만 생성 → 학습 데이터 누수 방지)
  out/normal_patches/{train,test}/*.png
  out/manifest.csv                        (파일, 원본 이미지, 분할, 종류)
  out/review_pseudo_normal_*.png         (눈으로 걸러낼 때 쓰는 원본|결과 비교 시트)

활용
  - PatchCore(anomalib Folder): normal_dir = pseudo_normal/train, 평가는 pseudo_normal/test(정상) + 평가용 NEU 원본(불량)
  - YOLO 배경 이미지: pseudo_normal/train 일부(학습셋의 5~10%)를 라벨 없이(빈 .txt) 학습 폴더에 추가 → 오검출 감소
  - 오검출률(양품을 불량으로 판정) 측정과 시연용 양품 흐름: pseudo_normal/test
  - 라벨에 없는 작은 결함이 남은 경우가 있으므로 review 시트를 보고 이상한 파일은 지울 것(보고서에는 '합성 정상'으로 표기)

필요 패키지: pip install opencv-python numpy
"""
import argparse, csv, hashlib, os, random, xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np

CLEAN_FOR_IMAGE = ['inclusion', 'scratches']
CLEAN_FOR_PATCH = ['inclusion', 'scratches', 'pitted_surface']
ALL_CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def find_pairs(root: Path):
    """(이름, 이미지 경로, 라벨 경로) 수집. 라벨은 VOC XML 또는 YOLO txt(labels 폴더) 모두 지원"""
    imgs, anns = {}, {}
    for d, _, files in os.walk(root, followlinks=True):
        for fn in files:
            p = Path(d) / fn
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                imgs[p.stem] = p
            elif p.suffix.lower() == '.xml' or (p.suffix.lower() == '.txt' and 'labels' in p.parts):
                anns[p.stem] = p
    return [(n, imgs[n], anns[n]) for n in sorted(imgs) if n in anns]


def read_boxes(xml_path, W=200, H=200):
    if Path(xml_path).suffix.lower() == '.txt':  # YOLO: cls cx cy w h (0~1)
        out = []
        for line in open(xml_path, encoding='utf-8'):
            v = line.split()
            if len(v) < 5:
                continue
            cx, cy, w, h = (float(t) for t in v[1:5])
            out.append((int(round((cx - w / 2) * W)), int(round((cy - h / 2) * H)),
                        int(round((cx + w / 2) * W)), int(round((cy + h / 2) * H))))
        return out
    r = ET.parse(xml_path).getroot()
    out = []
    for o in r.findall('object'):
        b = o.find('bndbox')
        out.append(tuple(int(float(b.find(k).text)) for k in ('xmin', 'ymin', 'xmax', 'ymax')))
    return out


def occupancy(boxes, margin, W, H):
    occ = np.zeros((H, W), np.int64)
    exp = []
    for x1, y1, x2, y2 in boxes:
        a = (max(0, x1 - margin), max(0, y1 - margin), min(W - 1, x2 + margin), min(H - 1, y2 + margin))
        exp.append(a)
        occ[a[1]:a[3] + 1, a[0]:a[2] + 1] = 1
    ii = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return exp, (lambda x, y, w, h: ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x] == 0)


def make_pseudo_normal(img, boxes, rng, margin=3):
    """결함 박스(여유 margin px)를 같은 이미지의 깨끗한 동일 크기 영역으로 대체. 불가능하면 None"""
    H, W = img.shape[:2]
    exp, free = occupancy(boxes, margin, W, H)
    out = img.copy()
    for (x1, y1, x2, y2) in exp:
        w, h = x2 - x1 + 1, y2 - y1 + 1
        cands = [(x, y) for y in range(1, H - h, 2) for x in range(1, W - w, 2) if free(x, y, w, h)]
        if not cands:
            return None
        ring = img[max(0, y1 - 6):min(H, y2 + 7), max(0, x1 - 6):min(W, x2 + 7)].mean()
        cands.sort(key=lambda p: abs(img[p[1]:p[1] + h, p[0]:p[0] + w].mean() - ring))
        dx, dy = rng.choice(cands[:max(1, len(cands) // 10)])
        donor = img[dy:dy + h, dx:dx + w]
        cx = min(max(x1 + w // 2, w // 2 + 1), W - (w - w // 2) - 1)
        cy = min(max(y1 + h // 2, h // 2 + 1), H - (h - h // 2) - 1)
        try:
            out = cv2.seamlessClone(donor, out, np.full((h, w), 255, np.uint8), (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:
            return None
    return out


def normal_patches(boxes, P, W, H, margin=4, stride=8):
    _, free = occupancy(boxes, margin, W, H)
    chosen = []
    for y in range(0, H - P + 1, stride):
        for x in range(0, W - P + 1, stride):
            if free(x, y, P, P) and all(abs(x - a) >= P or abs(y - b) >= P for a, b in chosen):
                chosen.append((x, y))
    return chosen


def is_test(name, test_set, ratio):
    if test_set is not None:
        return name in test_set
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 < ratio * 1000


def review_sheet(pairs, path, per_row=4):
    if not pairs:
        return
    tiles = [np.hstack([a, np.full((a.shape[0], 4, 3), 255, np.uint8), b]) for a, b in pairs]
    rows = []
    for k in range(0, len(tiles), per_row):
        row = tiles[k:k + per_row]
        while len(row) < per_row:
            row.append(np.full_like(tiles[0], 255))
        rows.append(np.hstack([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=255) for t in row]))
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--neu-root', required=True, help='NEU-DET 폴더 (VOC: IMAGES/ANNOTATIONS, YOLO: train|val/images,labels)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--test-list', help='평가용 원본 이미지 이름 목록 파일(없으면 이름 해시로 20%%)')
    ap.add_argument('--test-ratio', type=float, default=0.2)
    ap.add_argument('--patch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    out = Path(a.out)
    test_set = None
    if a.test_list:
        test_set = {Path(l.strip()).stem for l in open(a.test_list, encoding='utf-8') if l.strip()}
    for d in ('pseudo_normal/train', 'pseudo_normal/test', 'normal_patches/train', 'normal_patches/test'):
        (out / d).mkdir(parents=True, exist_ok=True)

    rows, review, stats = [], {'train': [], 'test': []}, {}
    for name, ip, xp in find_pairs(Path(a.neu_root)):
        cls = next((c for c in ALL_CLASSES if name.startswith(c)), None)
        if cls not in CLEAN_FOR_PATCH:
            continue
        img = cv2.imread(str(ip))
        H, W = img.shape[:2]
        boxes = read_boxes(xp, W, H)
        if test_set is None and any(part.lower() in ('val', 'valid', 'test') for part in Path(ip).parts):
            split = 'test'   # YOLO 폴더 구조면 val/test 이미지를 평가용으로 자동 사용
        elif test_set is None and any(part.lower() == 'train' for part in Path(ip).parts):
            split = 'train'
        else:
            split = 'test' if is_test(name, test_set, a.test_ratio) else 'train'
        s = stats.setdefault(cls, {'pseudo_normal': 0, 'patches': 0})

        if cls in CLEAN_FOR_IMAGE:
            pn = make_pseudo_normal(img, boxes, rng)
            if pn is not None:
                f = out / 'pseudo_normal' / split / f'{name}_pn.png'
                cv2.imwrite(str(f), cv2.cvtColor(pn, cv2.COLOR_BGR2GRAY))
                rows.append([f.relative_to(out).as_posix(), name, split, 'pseudo_normal'])
                s['pseudo_normal'] += 1
                vis = img.copy()
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                review[split].append((vis, pn))

        for k, (x, y) in enumerate(normal_patches(boxes, a.patch, W, H)):
            f = out / 'normal_patches' / split / f'{name}_p{k}.png'
            cv2.imwrite(str(f), cv2.cvtColor(img[y:y + a.patch, x:x + a.patch], cv2.COLOR_BGR2GRAY))
            rows.append([f.relative_to(out).as_posix(), name, split, f'patch@{x},{y}'])
            s['patches'] += 1

    with open(out / 'manifest.csv', 'w', newline='', encoding='utf-8') as fp:
        w = csv.writer(fp)
        w.writerow(['file', 'source_image', 'split', 'kind'])
        w.writerows(rows)
    for split, pairs in review.items():
        for k in range(0, len(pairs), 40):
            review_sheet(pairs[k:k + 40], out / f'review_pseudo_normal_{split}_{k // 40 + 1:02d}.png')

    print('class            pseudo_normal  patches')
    for c, s in stats.items():
        print(f"{c:16s} {s['pseudo_normal']:13d}  {s['patches']:7d}")
    print('split counts:', {sp: sum(1 for r in rows if r[2] == sp and r[3] == 'pseudo_normal') for sp in ('train', 'test')},
          '(pseudo_normal)')


if __name__ == '__main__':
    main()
