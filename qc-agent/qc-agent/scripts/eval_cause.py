"""원인 추적 정확도 평가 (시뮬레이션 시나리오 기준)
각 고장 에피소드 종료 시점에서 ① 급증 결함 탐지 ② SHAP 원인 순위 top-1/top-3 가 주입 정답과 일치하는지 확인.
※ 결과는 '시뮬레이션 시나리오 기준' 수치로만 보고할 것."""
import sys, sqlite3
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from agent.data import CONTEXT, VARS
from agent import tools as T

con = sqlite3.connect(ROOT / "data/factory.db")
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--early", type=int, default=0,
    help="에피소드 시작 후 N개 제품 시점에서 평가(조기 탐지 난이도). 0이면 에피소드 종료 시점")
A = ap.parse_args()
eps = con.execute("SELECT episode_id,start_seq,end_seq,var,direction,defect_type FROM fault_episodes").fetchall()
det = top1 = top3 = st1 = 0
print(f"{'ep':>3} {'정답 결함':<16}{'정답 변수':<18}{'탐지':<6}{'top-1':<18}{'판정'}")
for ep, s, e, var, dirn, d in eps:
    CONTEXT["now_seq"] = s + A.early if A.early else e
    win = min(300, CONTEXT["now_seq"] - s)
    spk = T.get_defect_summary(last_n=win)["spiking"]
    c = T.rank_causes(d, last_n=win)["candidates"]
    names = [x["var"] for x in c]
    ok1 = bool(names) and names[0] == var
    det += d in spk; top1 += ok1; top3 += var in names[:3]
    st1 += bool(names) and VARS[names[0]]["station"] == VARS[var]["station"]
    print(f"{ep:>3} {d:<16}{var:<18}{'O' if d in spk else 'X':<6}{(names[0] if names else '-'):<18}{'O' if ok1 else 'X'}")
n = len(eps)
print(f"\n급증 결함 탐지율 {det/n:.0%} | 원인 변수 top-1 {top1/n:.0%} | top-3 {top3/n:.0%} | 원인 공정 top-1 {st1/n:.0%}  (n={n}, 시뮬레이션 기준)")
