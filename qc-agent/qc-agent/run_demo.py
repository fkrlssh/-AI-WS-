"""시연 실행기
  python run_demo.py --scenario scratches            # 스크래치 급증 시나리오 재생
  python run_demo.py --scenario rolled-in_scale --offline   # API 없이 Fallback으로
  python run_demo.py approve 3 --note "가이드 교체 후 적용"  # 작업자 승인(Memory 기록)
"""
import argparse, sqlite3, sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
from agent.data import CONTEXT, KO
from agent import agent

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd")
ap.add_argument("--scenario", choices=list(KO), help="고장 에피소드를 골라 해당 시점으로 재생")
ap.add_argument("--episode", type=int, help="에피소드 번호 직접 지정")
ap.add_argument("--after", type=int, default=150, help="에피소드 시작 후 몇 개 제품 시점에서 실행할지")
ap.add_argument("--goal", default="최근 불량이 늘어난 것 같아. 원인 공정을 찾아서 조치 방안을 제안해줘.")
ap.add_argument("--offline", action="store_true", help="LLM 없이 Fallback 플래너로 실행")
p_ap = sub.add_parser("approve"); p_ap.add_argument("action_id", type=int); p_ap.add_argument("--note", default="")
p_rj = sub.add_parser("reject"); p_rj.add_argument("action_id", type=int); p_rj.add_argument("--note", default="")
a = ap.parse_args()

con = sqlite3.connect(ROOT / "data/factory.db")
if a.cmd in ("approve", "reject"):
    st = "approved" if a.cmd == "approve" else "rejected"
    con.execute("UPDATE actions SET status=?, note=? WHERE action_id=?", (st, a.note, a.action_id)); con.commit()
    print(f"action {a.action_id} → {st}"); sys.exit()

if a.scenario or a.episode:
    q = "SELECT episode_id,start_seq,end_seq,defect_type,var FROM fault_episodes WHERE " + \
        ("episode_id=?" if a.episode else "defect_type=? ORDER BY episode_id LIMIT 1")
    ep = con.execute(q, (a.episode or a.scenario,)).fetchone()
    CONTEXT["now_seq"] = min(ep[1] + a.after, ep[2])
    print(f"[시나리오] ep{ep[0]} {KO[ep[3]]} (주입 원인: {ep[4]}) / 현재 시점 seq={CONTEXT['now_seq']}")
agent.run(a.goal, offline=a.offline)
