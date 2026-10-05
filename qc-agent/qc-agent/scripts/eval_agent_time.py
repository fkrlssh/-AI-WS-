"""Agent 응답 시간 측정 — 제안서 목표 '조언 5초 이내'

  python scripts/eval_agent_time.py --n 10            # .env의 ANTHROPIC_API_KEY가 있으면 LLM 모드
  python scripts/eval_agent_time.py --n 10 --offline  # 규칙 기반 모드

infer.py로 결함 판정된 제품 n개를 고장 구간에서 고르고, 각 제품에 대해
"이 결함은 어느 공정에서 생겼고 어떻게 조치해야 하나?"를 물어 첫 답변까지 걸린 시간을 잰다.
측정 중 Agent가 조치안을 등록할 수 있으므로 factory.db를 백업했다가 끝나면 되돌린다.
"""
import argparse, json, shutil, sqlite3, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--offline", action="store_true")
ap.add_argument("--q", default="이 결함은 어느 공정에서 생겼고 어떻게 조치해야 하나?")
a = ap.parse_args()

DB = ROOT / "data/factory.db"
BAK = ROOT / "data/factory.db.bak"
shutil.copy(DB, BAK)
try:
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS detections(product_id TEXT PRIMARY KEY, seq INTEGER, ts TEXT, lot TEXT,
                 defect TEXT, conf REAL, boxes_json TEXT, image_path TEXT, saved_path TEXT, latency_ms REAL, detected_at TEXT)""")
    rows = con.execute("""SELECT p.product_id, p.seq, p.ts, p.lot, i.pred_defect, i.conf, i.image_path
        FROM products p JOIN inspections i USING(product_id)
        JOIN fault_episodes e ON p.seq BETWEEN e.start_seq AND e.end_seq
        WHERE i.pred_defect IS NOT NULL AND i.pred_defect != 'normal'
        ORDER BY p.seq""").fetchall()
    if not rows:
        raise SystemExit("판정된 결함 제품이 없습니다. simulate_process → infer.py 를 먼저 실행하세요.")
    step = max(1, len(rows) // a.n)
    pick = rows[::step][:a.n]
    con.executemany("INSERT OR IGNORE INTO detections VALUES (?,?,?,?,?,?,'[]',?,'',NULL,NULL)", pick)
    con.commit(); con.close()

    from agent import chat as C
    C.chat(pick[0][0], [], a.q, offline=True)  # 워밍업(모듈 로드·첫 SHAP 계산) — 앱 서버가 이미 떠 있는 상황과 맞춤
    times, modes = [], []
    for pid, *_ in pick:
        t0 = time.perf_counter()
        out = C.chat(pid, [], a.q, offline=a.offline)
        dt = time.perf_counter() - t0
        times.append(dt); modes.append(out["mode"])
        print(f"{pid} [{out['mode']}] {dt:.2f}s  도구 {len(out['steps'])}회")
    print(f"\n[Agent] {len(times)}건 평균 {statistics.mean(times):.2f}s · 최대 {max(times):.2f}s "
          f"· 5초 이내 {sum(t <= 5 for t in times)}/{len(times)} (모드: {set(modes)})")
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/eval_agent_time.json").write_text(json.dumps(
        {"n": len(times), "mode": sorted(set(modes)), "mean_s": round(statistics.mean(times), 2),
         "max_s": round(max(times), 2), "within_5s": sum(t <= 5 for t in times)}, ensure_ascii=False, indent=2), encoding="utf-8")
finally:
    shutil.copy(BAK, DB); BAK.unlink()
