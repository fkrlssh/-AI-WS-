"""
NEU-DET(YOLO 형식) 오프라인 데이터 증강
=========================================
학습(train) 데이터만 늘리고 검증(val)은 그대로 복사한다. 박스 좌표는 변환에 맞춰 함께 바뀐다.
OpenCV + NumPy만 사용한다(pip install opencv-python numpy).

1) 변형 증강   : 원본 1장당 k장(어려운 클래스는 --boost 배수만큼 더)
   - 기하: 좌우/상하 뒤집기, 확대 크롭(잘려 나간 결함은 40% 이상 보일 때만 라벨 유지)
   - 밝기: 밝기·대비, 감마, 조명 기울기(한쪽이 어두운 조명), CLAHE
   - 화질: 가우시안 노이즈, 블러/세로 방향 모션 블러(컨베이어 이동), JPEG 압축
   - 일부러 뺀 것: 90도 회전(압연 방향이라 줄무늬 결함 방향이 바뀜), 색조·채도(흑백 영상)
2) 결함 합성   : 의사 정상 배경(pn_data/pseudo_normal/train)에 inclusion·scratches 결함을 1~3개
                 자연스럽게 붙여 새 라벨 이미지를 만든다(--paste 장수)
3) 배경 이미지 : 의사 정상 이미지를 라벨 없이 학습셋의 일정 비율(--bg-ratio) 추가 → 오검출 감소

사용법
  python augment_data.py --src neu-det --out neu-det-aug --bg pn_data/pseudo_normal/train
  # 옵션: --k 3 --boost crazing=2,rolled-in_scale=2 --paste 300 --bg-ratio 0.08 --seed 0
  #       --yaml-root "C:/Users/1/Desktop/경남 AISW/neu-det-aug"   (data.yaml의 path를 직접 지정)

산출물
  out/train/{images,labels}  원본 + 증강 + 합성 + 배경
  out/val/{images,labels}    원본 그대로
  out/data.yaml              YOLO 학습용
  out/preview_augment.jpg    박스가 제대로 따라갔는지 눈으로 확인하는 시트
  out/augment_report.txt     클래스별 이미지·박스 수
"""
import argparse, math, random, shutil
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

DEFAULT_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
PASTE_CLASSES = ['inclusion', 'scratches']   # 배경(매끈한 표면)과 질감이 맞는 국소 결함만 합성
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
JPEG_Q = [cv2.IMWRITE_JPEG_QUALITY, 95]


# ---------------------------------------------------------------- 입출력
def read_names(yaml_path: Path):
    if not yaml_path.exists():
        return DEFAULT_NAMES
    names, in_names = [], False
    for ln in yaml_path.read_text(encoding='utf-8').splitlines():
        s = ln.strip()
        if s.startswith('names:'):
            rest = s[6:].strip()
            if rest.startswith('['):
                return [x.strip().strip('\'"') for x in rest.strip('[]').split(',') if x.strip()]
            in_names = True
            continue
        if in_names:
            if s.startswith('- '):
                names.append(s[2:].strip().strip('\'"'))
            elif ':' in s and s.split(':')[0].strip().isdigit():
                names.append(s.split(':', 1)[1].strip().strip('\'"'))
            elif s and not s.startswith('#'):
                break
    return names or DEFAULT_NAMES


def read_labels(p: Path, W, H):
    """YOLO txt → [[cls, x1, y1, x2, y2], ...] (픽셀)"""
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            v = line.split()
            if len(v) >= 5:
                c, cx, cy, w, h = int(v[0]), *map(float, v[1:5])
                out.append([c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(out, dtype=np.float64).reshape(-1, 5)


def write_labels(p: Path, boxes, W, H):
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, x1), min(float(W), x2)
        y1, y2 = max(0.0, y1), min(float(H), y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f'{int(c)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
    p.write_text('\n'.join(lines) + ('\n' if lines else ''))


def list_split(src: Path, split: str):
    idir, ldir = src / split / 'images', src / split / 'labels'
    if not idir.exists():
        return []
    return sorted((f, ldir / (f.stem + '.txt')) for f in idir.iterdir() if f.suffix.lower() in IMG_EXT)


# ---------------------------------------------------------------- 변형
def flip(img, b, horizontal):
    H, W = img.shape[:2]
    b = b.copy()
    if horizontal:
        img = img[:, ::-1]
        b[:, [1, 3]] = W - b[:, [3, 1]]
    else:
        img = img[::-1, :]
        b[:, [2, 4]] = H - b[:, [4, 2]]
    return np.ascontiguousarray(img), b


def zoom_crop(img, b, rng, smax=1.3, min_vis=0.4):
    """이미지 안쪽 창을 잘라 확대(테두리 채움 없음). 많이 잘린 결함은 라벨에서 제외"""
    H, W = img.shape[:2]
    s = rng.uniform(1.05, smax)
    cw, ch = W / s, H / s
    x0, y0 = rng.uniform(0, W - cw), rng.uniform(0, H - ch)
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    nb = b.copy()
    nb[:, [1, 3]] = (b[:, [1, 3]] - x0) * s
    nb[:, [2, 4]] = (b[:, [2, 4]] - y0) * s
    full = (nb[:, 3] - nb[:, 1]) * (nb[:, 4] - nb[:, 2])
    nb[:, [1, 3]] = nb[:, [1, 3]].clip(0, W)
    nb[:, [2, 4]] = nb[:, [2, 4]].clip(0, H)
    vis = (nb[:, 3] - nb[:, 1]) * (nb[:, 4] - nb[:, 2])
    keep = (vis >= min_vis * full) & (nb[:, 3] - nb[:, 1] >= 4) & (nb[:, 4] - nb[:, 2] >= 4)
    if not keep.any():
        return None, None
    return out, nb[keep]


def _sat(img):
    return float(np.mean((img <= 3) | (img >= 252)))


def _photometric(img, rng):
    H, W = img.shape[:2]
    f = img.astype(np.float32)
    if rng.random() < 0.8:                                   # 밝기·대비(평균 기준으로 조절해 쏠림 방지)
        m = f.mean()
        f = (f - m) * rng.uniform(0.75, 1.3) + m + rng.uniform(-25, 25)
    if rng.random() < 0.3:                                   # 감마
        f = 255.0 * np.power(np.clip(f, 0, 255) / 255.0, rng.uniform(0.7, 1.5))
    if rng.random() < 0.3:                                   # 조명 기울기
        a = rng.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        t = ((xx - W / 2) * math.cos(a) + (yy - H / 2) * math.sin(a)) / (max(W, H) / 2)
        f = f * (1 + rng.uniform(0.1, 0.25) * t)
    out = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:                                  # 국소 대비(CLAHE)
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)
    if rng.random() < 0.3:                                   # 센서 노이즈
        out = np.clip(out + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 8), out.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.2:                                   # 초점 흐림 / 컨베이어 이동 블러
        if rng.random() < 0.5:
            out = cv2.GaussianBlur(out, (3, 3), 0)
        else:
            k = rng.choice([3, 5])
            kernel = np.zeros((k, k), np.float32)
            kernel[:, k // 2] = 1.0 / k
            out = cv2.filter2D(out, -1, kernel)
    if rng.random() < 0.2:                                   # JPEG 압축 열화
        ok, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 90)])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out


def photometric(img, rng):
    """밝기 변형으로 하얗게/까맣게 날아가 결함이 사라지면(포화 픽셀 증가) 다시 뽑는다"""
    base = _sat(img)
    for _ in range(6):
        out = _photometric(img, rng)
        if _sat(out) <= base + 0.03:
            return out
    return img


def augment_once(img, b, rng):
    if rng.random() < 0.5:
        img, b = flip(img, b, True)
    if rng.random() < 0.5:
        img, b = flip(img, b, False)
    if rng.random() < 0.5:
        z_img, z_b = zoom_crop(img, b, rng)
        if z_img is not None:
            img, b = z_img, z_b
    return photometric(img, rng), b


# ---------------------------------------------------------------- 결함 합성
def build_bank(items, names, max_area=0.25):
    bank = []
    for ip, lp in items:
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        for c, x1, y1, x2, y2 in read_labels(lp, W, H):
            if names[int(c)] not in PASTE_CLASSES:
                continue
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))
            w, h = x2 - x1, y2 - y1
            if w < 6 or h < 6 or w * h > max_area * W * H:
                continue
            bank.append((int(c), img[y1:y2, x1:x2].copy()))
    return bank


def synthesize(bg, bank, rng, n_def, vis_min=12.0):
    """배경에 결함 조각을 붙인다. 붙인 뒤 원래 배경과 차이가 작아(눈에 안 보이는) 조각은 되돌리고 다른 조각으로 재시도"""
    H, W = bg.shape
    out = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    placed = []
    attempts = 0
    while len(placed) < n_def and attempts < n_def * 6:
        attempts += 1
        c, crop = rng.choice(bank)
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        if rng.random() < 0.5:
            crop = crop[::-1, :]
        s = rng.uniform(0.8, 1.2)
        w, h = max(6, int(crop.shape[1] * s)), max(6, int(crop.shape[0] * s))
        if w > W - 6 or h > H - 6:
            continue
        crop = cv2.resize(np.ascontiguousarray(crop), (w, h), interpolation=cv2.INTER_LINEAR)
        for _try in range(40):
            x, y = rng.randint(3, W - w - 3), rng.randint(3, H - h - 3)
            if all(x + w + 4 <= px1 or px2 + 4 <= x or y + h + 4 <= py1 or py2 + 4 <= y for _, px1, py1, px2, py2 in placed):
                break
        else:
            continue
        try:
            cand = cv2.seamlessClone(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), out, np.full((h, w), 255, np.uint8),
                                     (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)
        except cv2.error:
            continue
        diff = np.abs(cand[y:y + h, x:x + w, 0].astype(np.int16) - out[y:y + h, x:x + w, 0].astype(np.int16))
        if np.percentile(diff, 95) < vis_min:      # 거의 안 보이면 라벨 노이즈가 되므로 버림
            continue
        out = cand
        placed.append([c, x, y, x + w, y + h])
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), np.array(placed, dtype=np.float64).reshape(-1, 5)


def residual_score(gray):
    """조명 성분을 뺀 뒤 국소적으로 튀는 밝기(남은 긁힘·반점)의 세기. 클수록 잔여 결함 의심"""
    f = gray.astype(np.float32)
    r = cv2.GaussianBlur(f - cv2.GaussianBlur(f, (0, 0), 8), (0, 0), 1.5)
    mad = np.median(np.abs(r - np.median(r))) + 1e-6
    return float(np.percentile(np.abs(r), 99.95) / mad)


# ---------------------------------------------------------------- 미리보기
COLORS = [(66, 135, 245), (60, 180, 75), (245, 130, 48), (145, 30, 180), (230, 25, 75), (0, 170, 170)]


def draw(img, b, names, title):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for c, x1, y1, x2, y2 in b:
        col = COLORS[int(c) % len(COLORS)]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2) - 1, int(y2) - 1), col, 1)
        cv2.putText(vis, names[int(c)][:3], (int(x1) + 2, max(10, int(y1) + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    vis = cv2.copyMakeBorder(vis, 0, 16, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(vis, title[:30], (2, vis.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def save_grid(tiles, path, cols=6):
    if not tiles:
        return
    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    grid = np.full((rows * (h + 4), cols * (w + 4), 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * (h + 4):r * (h + 4) + t.shape[0], c * (w + 4):c * (w + 4) + t.shape[1]] = t
    cv2.imwrite(str(path), grid, JPEG_Q)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='NEU-DET YOLO 데이터 오프라인 증강')
    ap.add_argument('--src', required=True, help='원본 YOLO 데이터 폴더(train/val, images/labels)')
    ap.add_argument('--out', required=True, help='출력 폴더(새로 생성)')
    ap.add_argument('--k', type=int, default=3, help='원본 1장당 변형 증강 장수')
    ap.add_argument('--boost', default='crazing=2,rolled-in_scale=2', help='클래스별 추가 배수(인식이 어려운 클래스)')
    ap.add_argument('--bg', help='의사 정상 이미지 폴더(결함 합성 배경 + 배경 이미지)')
    ap.add_argument('--paste', type=int, default=300, help='결함 합성 이미지 장수(--bg 필요)')
    ap.add_argument('--bg-ratio', type=float, default=0.05, help='라벨 없는 배경 이미지 비율(최종 학습셋 대비)')
    ap.add_argument('--bg-drop', type=float, default=0.15, help='잔여 결함 의심 점수 상위 비율만큼 배경 후보에서 제외')
    ap.add_argument('--yaml-root', help='data.yaml의 path 값(기본: 출력 폴더 절대경로)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true', help='중간에 끊긴 경우 이어서 실행(이미 있는 증강 파일은 다시 쓰지 않음)')
    a = ap.parse_args()

    rng = random.Random(a.seed)
    src, out = Path(a.src), Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise SystemExit(f'출력 폴더가 비어 있지 않습니다: {out} (다른 이름을 쓰거나 지운 뒤 실행)')
    names = read_names(src / 'data.yaml')
    boost = {}
    for kv in filter(None, (a.boost or '').split(',')):
        k, v = kv.split('=')
        boost[k.strip()] = int(v)

    for sp in ('train', 'val', 'test'):
        if (src / sp / 'images').exists():
            (out / sp / 'images').mkdir(parents=True, exist_ok=True)
            (out / sp / 'labels').mkdir(parents=True, exist_ok=True)

    # 검증/테스트는 그대로 복사(증강 금지)
    for sp in ('val', 'test'):
        for ip, lp in list_split(src, sp):
            shutil.copy2(ip, out / sp / 'images' / ip.name)
            if lp.exists():
                shutil.copy2(lp, out / sp / 'labels' / lp.name)

    train = list_split(src, 'train')
    ti, tl = out / 'train' / 'images', out / 'train' / 'labels'
    report = Counter()
    box_before, box_after = Counter(), Counter()
    preview = []
    per_class_prev = Counter()

    # 1) 원본 + 변형 증강
    for ip, lp in train:
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        b = read_labels(lp, W, H)
        if not (a.resume and (ti / ip.name).exists() and (tl / (ip.stem + '.txt')).exists()):
            shutil.copy2(ip, ti / ip.name)
            write_labels(tl / (ip.stem + '.txt'), b, W, H)
        report['원본'] += 1
        for c in b[:, 0]:
            box_before[names[int(c)]] += 1
            box_after[names[int(c)]] += 1
        if len(b) == 0:
            continue
        main_cls = names[Counter(b[:, 0].astype(int).tolist()).most_common(1)[0][0]]
        n_var = a.k * boost.get(main_cls, 1)
        for j in range(n_var):
            aug_img, aug_b = augment_once(img, b, rng)
            stem = f'{ip.stem}_aug{j}'
            if not (a.resume and (ti / f'{stem}.jpg').exists() and (tl / f'{stem}.txt').exists()):
                cv2.imwrite(str(ti / f'{stem}.jpg'), aug_img, JPEG_Q)
                write_labels(tl / f'{stem}.txt', aug_b, W, H)
            report['변형 증강'] += 1
            for c in aug_b[:, 0]:
                box_after[names[int(c)]] += 1
            if per_class_prev[main_cls] < 3 and rng.random() < 0.02:
                per_class_prev[main_cls] += 1
                preview.append(draw(aug_img, aug_b, names, stem))

    # 2) 결함 합성 + 3) 배경 이미지
    bgs = []
    if a.bg:
        bgs = sorted(p for p in Path(a.bg).iterdir() if p.suffix.lower() in IMG_EXT)
        scored = sorted(bgs, key=lambda p: residual_score(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)))
        n_keep = len(scored) - int(round(len(scored) * a.bg_drop))
        dropped = scored[n_keep:]
        bgs = sorted(scored[:n_keep])
        print(f'배경 후보 {len(scored)}장 중 잔여 결함 의심 {len(dropped)}장 제외 → {len(bgs)}장 사용')
        (out / 'bg_excluded.txt').write_text('\n'.join(p.name for p in dropped) + '\n', encoding='utf-8')
    if bgs and a.paste > 0:
        bank = build_bank(train, names)
        print(f'합성용 결함 조각: {len(bank)}개 ({", ".join(PASTE_CLASSES)})')
        for i in range(a.paste):
            bg = cv2.imread(str(rng.choice(bgs)), cv2.IMREAD_GRAYSCALE)
            if rng.random() < 0.5:
                bg = np.ascontiguousarray(bg[:, ::-1])
            img, b = synthesize(bg, bank, rng, rng.randint(1, 3))
            if len(b) == 0:
                continue
            img = photometric(img, rng)
            stem = f'synth_{i:04d}'
            cv2.imwrite(str(ti / f'{stem}.jpg'), img, JPEG_Q)
            write_labels(tl / f'{stem}.txt', b, img.shape[1], img.shape[0])
            report['결함 합성'] += 1
            for c in b[:, 0]:
                box_after[names[int(c)]] += 1
            if per_class_prev['synth'] < 6 and rng.random() < 0.05:
                per_class_prev['synth'] += 1
                preview.append(draw(img, b, names, stem))
    if bgs and a.bg_ratio > 0:
        n_now = sum(report.values())
        n_bg = round(a.bg_ratio * n_now / (1 - a.bg_ratio))
        for i in range(n_bg):
            img = cv2.imread(str(bgs[i % len(bgs)]), cv2.IMREAD_GRAYSCALE)
            img, _ = augment_once(img, np.zeros((0, 5)), rng) if i >= len(bgs) else (img, None)
            stem = f'bg_{i:04d}'
            cv2.imwrite(str(ti / f'{stem}.jpg'), img, JPEG_Q)
            (tl / f'{stem}.txt').write_text('')
            report['배경(라벨 없음)'] += 1

    # data.yaml
    root = a.yaml_root or out.resolve().as_posix()
    yaml = [f'# 증강 데이터셋 (augment_data.py). 다른 PC에서 학습하면 path를 그 위치로 바꾸세요', f'path: {root}',
            'train: train/images', 'val: val/images']
    if (out / 'test' / 'images').exists():
        yaml.append('test: test/images')
    yaml += [f'nc: {len(names)}', 'names:'] + [f'- {n}' for n in names]
    (out / 'data.yaml').write_text('\n'.join(yaml) + '\n', encoding='utf-8')

    save_grid(preview, out / 'preview_augment.jpg')
    lines = [f'학습 이미지: {sum(report.values())}장 (원본 {len(train)}장 → {sum(report.values()) / max(1, len(train)):.1f}배)']
    lines += [f'  {k}: {v}' for k, v in report.items()]
    lines += ['클래스별 박스 수 (원본 → 증강 후)']
    lines += [f'  {n:16s} {box_before[n]:6d} → {box_after[n]:6d}' for n in names]
    lines += [f'검증 이미지: {len(list_split(out, "val"))}장 (증강하지 않음)']
    (out / 'augment_report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
