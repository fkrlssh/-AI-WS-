"""Agent 응답 시간 측정 — 제안서 목표 '불량 발생 후 조언 출력까지 5초 이내'

  python scripts/eval_agent_time.py --provider offline --n 10   # 규칙·통계 엔진만
  python scripts/eval_agent_time.py --provider ollama  --n 10   # 로컬 LLM (근거 고정형)
  python scripts/eval_agent_time.py --provider claude  --n 10   # Claude API (.env 키 필요)

두 가지 시간을 따로 잰다.
- 조언 시간(first_advice): 질문 후 '검증된 조언'(원인·조정값·SOP 근거)이 화면에 나오기까지.
  앱은 이 시점에 엔진 조언을 먼저 표시한다. 제안서 목표(5초)와 비교하는 값.
- 전체 시간(total): LLM 자연어 설명까지 끝나기까지.
표본: infer.py로 불량 판정된 고장 구간 제품 n개 × 질문 3개(원인·조치·이력), 질문마다 새 대화.
측정 전 워밍업 1회(모델·SHAP 로드), 측정 중 등록된 조치가 남지 않도록 factory.db를 백업했다가 되돌린다.
결과: runs/eval_agent_time.json
"""
import argparse, json, platform, shutil, sqlite3, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["offline", "ollama", "claude"], default="offline")
ap.add_argument("--n", type=int, default=10, help="제품 수 (질문은 제품당 3개)")
a = ap.parse_args()
QS = ["이 결함은 어느 공정에서 발생했어?", "어떻게 조치해야 해?", "과거에 비슷한 조치가 있었어?"]

DB, BAK = ROOT / "data/factory.db", ROOT / "data/factory.db.timebak"
shutil.copy(DB, BAK)
try:
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS detections(product_id TEXT PRIMARY KEY, seq INTEGER, ts TEXT, lot TEXT,
                 defect TEXT, conf REAL, boxes_json TEXT, image_path TEXT, saved_path TEXT, latency_ms REAL, detected_at TEXT)""")
    rows = con.execute("""SELECT p.product_id, p.seq, p.ts, p.lot, i.pred_defect, i.conf, i.image_path
        FROM products p JOIN inspections i USING(product_id)
        JOIN fault_episodes e ON p.seq BETWEEN e.start_seq AND e.end_seq
        WHERE i.pred_defect IS NOT NULL AND i.pred_defect != 'normal' ORDER BY p.seq""").fetchall()
    if not rows:
        raise SystemExit("판정된 결함 제품이 없습니다. simulate_process → infer.py 를 먼저 실행하세요.")
    step = max(1, len(rows) // a.n)
    pick = rows[::step][:a.n]
    con.executemany("INSERT OR REPLACE INTO detections VALUES (?,?,?,?,?,?,'[]',?,'',NULL,NULL)", pick)
    con.commit(); con.close()

    from agent import chat as C
    prov = None if a.provider == "offline" else a.provider
    C.chat(pick[0][0], [], QS[0], offline=prov is None, provider=prov)   # 워밍업 (측정 제외)

    first, total, modes = [], [], []
    for pid, *_ in pick:
        for q in QS:
            r = C.chat(pid, [], q, offline=prov is None, provider=prov)
            t = r["timing"]
            first.append(t["first_advice"]); total.append(t["total"]); modes.append(r["mode"])
            print(f"{pid} {q[:12]:<12} [{r['mode']}] 조언 {t['first_advice']:5.2f}s / 전체 {t['total']:6.2f}s")

    def stat(xs):
        return {"mean": round(statistics.mean(xs), 2), "median": round(statistics.median(xs), 2),
                "max": round(max(xs), 2), "within_5s": sum(x <= 5 for x in xs), "n": len(xs)}
    res = {"provider": a.provider, "device": platform.platform(), "modes": sorted(set(modes)),
           "first_advice": stat(first), "total": stat(total)}
    f, tt = res["first_advice"], res["total"]
    print(f"\n[{a.provider}] {f['n']}건")
    print(f"  조언 시간  평균 {f['mean']}s · 중앙 {f['median']}s · 최대 {f['max']}s · 5초 이내 {f['within_5s']}/{f['n']}")
    print(f"  전체 시간  평균 {tt['mean']}s · 중앙 {tt['median']}s · 최대 {tt['max']}s · 5초 이내 {tt['within_5s']}/{tt['n']}")
    (ROOT / "runs").mkdir(exist_ok=True)
    out = ROOT / f"runs/eval_agent_time_{a.provider}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out.relative_to(ROOT)}")
finally:
    shutil.copy(BAK, DB); BAK.unlink()
