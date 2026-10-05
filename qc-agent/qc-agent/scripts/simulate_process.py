"""열연 공정 이력 시뮬레이터 → data/factory.db (SQLite)

- 제품(코일)마다 제품ID·타임스탬프·공정변수 이력을 생성
- '고장 에피소드'를 주입: 특정 변수가 서서히 정상범위를 벗어나고, 연관 결함 확률이 올라감
- 결함 제품에는 NEU-DET val 이미지를 결함 유형에 맞춰 배정 (학습에 안 쓴 이미지)
- fault_episodes 테이블 = 원인 추적 정확도 평가용 정답지

※ 모든 공정 데이터는 시뮬레이션입니다. (docs/SOURCES.md, 보고서에 명시)
"""
import argparse, json, sqlite3, random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = yaml.safe_load(open(ROOT / "config/process_spec.yaml", encoding="utf-8"))
VARS = SPEC["variables"]
STATION_OFFSET_MIN = {"STEELMAKING": -300, "REHEAT": -60, "DESCALE": -8, "FINISH": -5, "COOL": -2}

SCHEMA = """
DROP TABLE IF EXISTS products; DROP TABLE IF EXISTS process_log; DROP TABLE IF EXISTS inspections;
DROP TABLE IF EXISTS fault_episodes; DROP TABLE IF EXISTS actions; DROP TABLE IF EXISTS detections;
CREATE TABLE products(product_id TEXT PRIMARY KEY, seq INTEGER, ts TEXT, lot TEXT);
CREATE TABLE process_log(product_id TEXT, station TEXT, var TEXT, value REAL, ts TEXT);
CREATE INDEX idx_pl ON process_log(product_id);
CREATE TABLE inspections(product_id TEXT PRIMARY KEY, ts TEXT, image_path TEXT,
    gt_defect TEXT, pred_defect TEXT, conf REAL, bbox_json TEXT, source TEXT);
CREATE TABLE fault_episodes(episode_id INTEGER, start_seq INTEGER, end_seq INTEGER,
    var TEXT, direction TEXT, defect_type TEXT, shift_std REAL);
CREATE TABLE actions(action_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, defect_type TEXT,
    var TEXT, from_value REAL, to_value REAL, status TEXT, reason TEXT,
    rate_before REAL, rate_after REAL, note TEXT);
"""


def build_episodes(n, rng, per_type=2):
    causes = [(d, c["var"], c["direction"]) for d, cs in SPEC["defect_causes"].items() for c in cs]
    # 결함 유형별 per_type개씩, 원인 변수는 해당 유형 후보 중에서 번갈아 선택
    plan = []
    for d, cs in SPEC["defect_causes"].items():
        for k in range(per_type):
            c = cs[k % len(cs)]
            plan.append((d, c["var"], c["direction"]))
    rng.shuffle(plan)
    eps, seq = [], 300                      # 앞 300개는 정상 워밍업
    gap = (n - 300) // len(plan)
    for i, (d, v, dirn) in enumerate(plan):
        length = rng.randint(int(gap * 0.25), int(gap * 0.40))
        start = seq + rng.randint(20, max(21, gap - length - 20))
        eps.append(dict(episode_id=i + 1, start_seq=start, end_seq=start + length,
                        var=v, direction=dirn, defect_type=d, shift_std=round(rng.uniform(3.5, 5.0), 2)))
        seq += gap
    return eps


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(n=6000, seed=42, base_rate=0.02):
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    eps = build_episodes(n, rng)

    # 공정변수 생성: 평균 + AR(1) 느린 흔들림 + 측정 노이즈
    data = {}
    for v, s in VARS.items():
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.98 * ar[t - 1] + nrng.normal(0, 0.2)
        data[v] = s["mean"] + s["std"] * (0.5 * ar / (ar.std() + 1e-9) + nrng.normal(0, 1, n))

    # 고장 주입: 에피소드 구간에서 변수를 서서히(앞 30%) 이탈 → 유지 → 끝나면 복귀
    for e in eps:
        s, t0, t1 = VARS[e["var"]], e["start_seq"], e["end_seq"]
        L, ramp = t1 - t0, max(1, int((t1 - t0) * 0.3))
        sign = -1 if e["direction"] == "low" else 1
        prof = np.minimum(np.arange(L) / ramp, 1.0)
        data[e["var"]][t0:t1] += sign * e["shift_std"] * s["std"] * prof

    # 결함 확률: 기본률 + 원인 변수 이탈 정도(z)에 따른 증가분
    classes = SPEC["classes"]
    probs = np.full((n, len(classes)), base_rate / len(classes))
    for j, d in enumerate(classes):
        for c in SPEC["defect_causes"][d]:
            s = VARS[c["var"]]
            z = (data[c["var"]] - s["mean"]) / s["std"] * (-1 if c["direction"] == "low" else 1)
            probs[:, j] += 0.35 * sigmoid((z - 3.0) * 2.5)

    # 이미지 풀 (val split, 결함 유형별)
    img_dir = ROOT / "data/neu-det/val/images"
    pool = {d: sorted(str(p.relative_to(ROOT)) for p in img_dir.glob(f"{d}_*.jpg")) for d in classes}
    idx = {d: 0 for d in classes}

    db = sqlite3.connect(ROOT / "data/factory.db")
    db.executescript(SCHEMA)
    t = datetime(2026, 10, 1, 6, 0, 0)
    prod_rows, log_rows, ins_rows = [], [], []
    for i in range(n):
        t += timedelta(seconds=rng.randint(45, 75))
        pid = f"C{t:%y%m%d}-{i:05d}"
        prod_rows.append((pid, i, t.isoformat(), f"L{i // 50:04d}"))
        for v, s in VARS.items():
            ts = (t + timedelta(minutes=STATION_OFFSET_MIN[s["station"]])).isoformat()
            log_rows.append((pid, s["station"], v, round(float(data[v][i]), 3), ts))
        # 결함 샘플링: 각 유형 독립 시행 후 가장 확률 높은 것 1개만 채택
        hits = [j for j in range(len(classes)) if rng.random() < probs[i, j]]
        gt = None
        if hits:
            gt = classes[max(hits, key=lambda j: probs[i, j])]
        img = None
        if gt and pool[gt]:
            img = pool[gt][idx[gt] % len(pool[gt])]; idx[gt] += 1
        ins_rows.append((pid, t.isoformat(), img, gt or "normal", None, None, None, None))

    db.executemany("INSERT INTO products VALUES (?,?,?,?)", prod_rows)
    db.executemany("INSERT INTO process_log VALUES (?,?,?,?,?)", log_rows)
    db.executemany("INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?)", ins_rows)
    db.executemany("INSERT INTO fault_episodes VALUES (:episode_id,:start_seq,:end_seq,:var,:direction,:defect_type,:shift_std)", eps)
    db.commit()

    n_def = sum(1 for r in ins_rows if r[3] != "normal")
    print(f"[done] products={n}, defects={n_def} ({n_def/n:.1%}), episodes={len(eps)} → data/factory.db")
    for e in eps:
        print(f"  ep{e['episode_id']:>2} seq {e['start_seq']:>4}-{e['end_seq']:<4} {e['defect_type']:<16} ← {e['var']} {e['direction']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.n, a.seed)
