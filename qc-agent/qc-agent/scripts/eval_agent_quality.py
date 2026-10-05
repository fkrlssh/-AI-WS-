"""Agent 답변 품질 자동 평가 — 여러 제품 × 질문으로 돌려 보고 문제 유형별로 집계

  python scripts/eval_agent_quality.py --provider offline          # 규칙 엔진만 (수 초)
  python scripts/eval_agent_quality.py --provider ollama --n 24    # 로컬 LLM (제품당 수십 초)
  python scripts/eval_agent_quality.py --provider claude --n 24    # Claude API (.env 키 필요, 약 $2)

표본: infer.py로 불량 판정된 제품 중 ① 고장 구간 안(진짜 공정 문제) ② 고장 구간 밖(단발성·오검출)을 반반,
      결함 유형이 고르게 섞이도록 뽑는다. 질문 3개(원인·조치·이력)를 각각 새 대화로 묻는다.

자동 검사 항목 (문제가 있으면 플래그)
- fallback   : LLM 답이 검증에 실패해 엔진 원문으로 대체됨 (LLM 모드에서만 의미 있음)
- 누락       : 이 제품의 정상범위 이탈 변수를 답에서 언급하지 않음 (원인 질문)
- 모순       : 이탈 변수를 '정상(범위 안/내)'이라고 말함
- 지시문노출 : [분석 결과] 같은 내부 자료 이름이 답에 나옴
- 오조치     : 고장 구간 밖(단발성) 제품인데 조정값을 제안함
- 미조치     : 고장 구간 안 제품인데 조정·관찰 판단 없이 끝남
- 느림       : 응답 5초 초과
결과: runs/eval_agent_quality.md (사람이 읽는 표) + .csv (전체 답변)
※ 측정 중 등록된 조치가 남지 않도록 factory.db를 백업했다가 끝나면 되돌린다.
"""
import argparse, csv, random, re, shutil, sqlite3, sys, time
from collections import Counter
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
ap.add_argument("--n", type=int, default=24, help="제품 수 (질문은 제품당 3개)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
QS = {"원인": "이 결함은 어느 공정에서 발생했어?", "조치": "어떻게 조치해야 해?", "이력": "과거에 비슷한 조치가 있었어?"}

DB, BAK = ROOT / "data/factory.db", ROOT / "data/factory.db.evalbak"
shutil.copy(DB, BAK)
try:
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS detections(product_id TEXT PRIMARY KEY, seq INTEGER, ts TEXT, lot TEXT,
                 defect TEXT, conf REAL, boxes_json TEXT, image_path TEXT, saved_path TEXT, latency_ms REAL, detected_at TEXT)""")
    rows = con.execute("""SELECT p.product_id, p.seq, p.ts, p.lot, i.pred_defect, i.conf, i.image_path,
            EXISTS(SELECT 1 FROM fault_episodes e WHERE p.seq BETWEEN e.start_seq AND e.end_seq AND e.defect_type = i.pred_defect)
        FROM products p JOIN inspections i USING(product_id)
        WHERE i.pred_defect IS NOT NULL AND i.pred_defect != 'normal' AND p.seq > 300 ORDER BY p.seq""").fetchall()
    rnd = random.Random(a.seed)
    pick = []
    for in_ep in (1, 0):   # 고장 구간 안/밖 반반, 결함 유형 고르게
        pool = [r for r in rows if r[7] == in_ep]
        by = {}
        for r in pool:
            by.setdefault(r[4], []).append(r)
        k = a.n // 2
        while k > 0 and any(by.values()):
            for d in sorted(by):
                if by[d] and k > 0:
                    pick.append(by[d].pop(rnd.randrange(len(by[d])))); k -= 1
    con.executemany("INSERT OR REPLACE INTO detections VALUES (?,?,?,?,?,?,'[]',?,'',NULL,NULL)", [r[:7] for r in pick])
    con.commit(); con.close()

    from agent import chat as C
    from agent.data import KO
    prov = None if a.provider == "offline" else a.provider
    C.chat(pick[0][0], [], QS["원인"], offline=True)   # 워밍업
    out, flags = [], Counter()
    for r in pick:
        pid, in_ep = r[0], bool(r[7])
        oor = [t for t in C.get_product_trace(pid)["trace"] if t["out_of_range"]]
        for qk, q in QS.items():
            t0 = time.perf_counter()
            res = C.chat(pid, [], q, offline=prov is None, provider=prov)
            dt = time.perf_counter() - t0
            txt, f = res["text"], []
            if prov and ("엔진 원문" in res["mode"] or res["mode"] == "offline"):
                f.append("fallback")
            if qk == "원인":
                if any(o["desc"] not in txt for o in oor):
                    f.append("누락")
            for o in oor:
                for sent in re.split(r"(?<=[.\n])", txt):
                    if o["desc"] in sent and re.search(r"정상\s*(범위\s*)?(안|내|에\s*있)", sent) and "벗어" not in sent \
                            and "이탈" not in sent:
                        f.append("모순"); break
            if re.search(r"\[(분석 결과|Agent 추가 조사)\]|지시문", txt):
                f.append("지시문노출")
            if qk == "조치":
                proposes = "[조치 제안]" in txt or bool(re.search(r"→\s*\**\d", txt))
                if not in_ep and proposes:
                    f.append("오조치")
                if in_ep and not (proposes or "관찰" in txt or "점검" in txt or "통보" in txt):
                    f.append("미조치")
            if dt > 5:
                f.append("느림")
            flags.update(f)
            out.append({"product": pid, "defect": KO.get(r[4], r[4]), "conf": r[5], "고장구간": "안" if in_ep else "밖",
                        "이탈": len(oor), "질문": qk, "mode": res["mode"], "sec": round(dt, 2),
                        "flags": ",".join(f), "answer": txt.replace("\n", " ")})
            print(f"{pid} {qk} [{res['mode']}] {dt:5.1f}s {','.join(f) or 'OK'}")

    (ROOT / "runs").mkdir(exist_ok=True)
    with open(ROOT / "runs/eval_agent_quality.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    n = len(out)
    ok = sum(1 for o in out if not o["flags"])
    md = [f"# Agent 답변 품질 평가 ({a.provider}, 제품 {len(pick)}개 × 질문 3개 = {n}건)", "",
          f"- 문제 없음: **{ok}/{n} ({ok / n:.0%})**",
          f"- 평균 응답 {sum(o['sec'] for o in out) / n:.2f}s · 최대 {max(o['sec'] for o in out):.2f}s", "",
          "| 문제 유형 | 건수 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in flags.most_common()] + \
         ["", "## 문제 사례", "", "| 제품 | 결함 | 구간 | 질문 | 문제 | 답변(앞부분) |", "|---|---|---|---|---|---|"] + \
         [f"| {o['product']} | {o['defect']} | {o['고장구간']} | {o['질문']} | {o['flags']} | {o['answer'][:120]} |"
          for o in out if o["flags"]]
    (ROOT / "runs/eval_agent_quality.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md[:8 + len(flags)]))
    print("\n[saved] runs/eval_agent_quality.md, runs/eval_agent_quality.csv")
finally:
    shutil.copy(BAK, DB); BAK.unlink()
