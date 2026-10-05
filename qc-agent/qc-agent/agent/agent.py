"""품질 원인추적 Agent
- 기본: LangChain + Claude가 스스로 계획을 세우고 도구를 골라 호출 (tool calling 루프)
- Fallback: API 키가 없거나 호출 실패 시, 같은 도구를 정해진 순서로 실행하는 규칙 기반 플래너
모든 단계는 logs/*.jsonl 실행 로그로 남아 대시보드·시연에서 '판단 과정'을 보여줄 수 있다."""
import json, os, time
from datetime import datetime
from pathlib import Path
from . import tools as T
from .data import KO

ROOT = Path(__file__).resolve().parents[1]
MAX_STEPS = 12

SYSTEM = """당신은 열연 강판 공장의 품질 원인추적 AI Agent입니다.
목표: 결함 급증을 감지하면 원인 공정·변수를 데이터로 규명하고, 안전한 조정값을 근거와 함께 제안합니다.

규칙
1. 첫 응답에서 도구를 부르기 전에, 무엇을 어떤 순서로 확인할지 한국어로 3~6줄 계획을 먼저 밝히세요.
2. 수치는 도구 결과에 있는 값만 쓰세요. 추측으로 숫자를 만들지 마세요.
3. 원인 후보는 rank_causes(SHAP)와 get_process_history/detect_deviation 결과가 서로 맞는지 교차 확인하세요.
4. 원인 변수의 adjustable이 false이면 조정하지 말고 상류 공정 통보를 제안하세요.
5. 조정값은 search_sop의 기준을 참고해 정하고, 반드시 check_adjustment로 검증하세요.
   검증 실패 시 safe_value로 다시 검증한 뒤 사용하세요. (목표값이 멀면 1차 조정값만 제안)
6. get_action_history로 과거 조치 이력을 확인하고, 같은 조치가 거절·실패한 적이 있으면 반영하세요.
7. 마지막으로 submit_recommendation으로 제안을 '승인 대기' 등록하세요. 설비를 직접 바꾸지 않습니다.
8. 최종 답변 형식:
   [상황] 결함 유형·발생률 변화
   [원인] 순위와 근거 수치(SHAP 기여도, 정상 대비 이탈)
   [조치 제안] 공정·변수, 현재값 → 제안값, SOP 근거, 기대효과
   [주의] 불확실성과 확인 필요 사항"""


class RunLog:
    def __init__(self, goal, mode):
        (ROOT / "logs").mkdir(exist_ok=True)
        self.path = ROOT / "logs" / f"run_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self.t0 = time.time()
        self.write("goal", {"goal": goal, "mode": mode})

    def write(self, kind, payload):
        rec = {"t": round(time.time() - self.t0, 2), "kind": kind, **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._print(kind, payload)

    @staticmethod
    def _print(kind, p):
        if kind == "goal":   print(f"\n🎯 목표: {p['goal']}  (mode={p['mode']})")
        elif kind == "think": print(f"\n🧠 {p['text'].strip()}")
        elif kind == "tool_call": print(f"🔧 {p['name']}({json.dumps(p['args'], ensure_ascii=False)})")
        elif kind == "tool_result": print(f"   ↳ {p['brief']}")
        elif kind == "final": print(f"\n✅ 최종 결과\n{p['text']}")
        elif kind == "error": print(f"⚠️  {p['text']}")


def _brief(name, r):
    try:
        if name == "get_defect_summary":
            t = r["by_type"][0]; return f"급증={r['spiking']} / {t['name_ko']} {t['baseline_rate']:.1%}→{t['rate']:.1%}"
        if name == "detect_deviation": return f"이탈 변수={r['abnormal']}"
        if name == "rank_causes": return " > ".join(f"{c['var']}({c['shap_share']:.0%},{c['direction']})" for c in r["candidates"][:3]) or r.get("error")
        if name == "get_process_history": v = r["variables"][0]; return f"결함 {r['n_defect']}건, 최대 차이 {v['var']} {v['gap_std']:+.1f}σ"
        if name == "search_sop": return " | ".join(x["section"] for x in r["results"])
        if name == "check_adjustment": return f"ok={r.get('ok')} safe={r.get('safe_value')} {r.get('issues')}"
        if name == "get_action_history": return f"이력 {len(r['history'])}건"
        if name == "submit_recommendation": return f"등록={r['registered']} id={r.get('action_id')}"
    except Exception:
        pass
    return str(r)[:120]


def _call(log, name, args):
    log.write("tool_call", {"name": name, "args": args})
    fn = {f.__name__: f for f in T.TOOLS}[name]
    try:
        r = fn(**args)
    except Exception as e:
        r = {"error": f"{type(e).__name__}: {e}"}
    log.write("tool_result", {"name": name, "brief": _brief(name, r), "result": r})
    return r


# ---------------- Claude(LangChain) Agent ----------------
def run_llm(goal: str, log: RunLog) -> str:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import StructuredTool

    lc_tools = [StructuredTool.from_function(f) for f in T.TOOLS]
    llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
                        temperature=0, max_tokens=2000).bind_tools(lc_tools)
    msgs = [SystemMessage(SYSTEM), HumanMessage(goal)]
    for _ in range(MAX_STEPS):
        ai = llm.invoke(msgs)
        msgs.append(ai)
        text = ai.text if isinstance(getattr(ai, "text", None), str) else str(ai.content)
        if not ai.tool_calls:
            log.write("final", {"text": text}); return text
        if text.strip():
            log.write("think", {"text": text})
        for tc in ai.tool_calls:
            r = _call(log, tc["name"], tc["args"])
            msgs.append(ToolMessage(json.dumps(r, ensure_ascii=False, default=str), tool_call_id=tc["id"]))
    log.write("error", {"text": "최대 단계 초과"}); return "최대 단계를 초과했습니다."


# ---------------- Fallback 규칙 기반 플래너 ----------------
def run_fallback(goal: str, log: RunLog) -> str:
    log.write("think", {"text": "계획: ①결함 급증 확인 → ②공정변수 이탈 확인 → ③결함-공정이력 결합 비교 → "
                                "④SHAP 원인 순위 → ⑤과거 조치 이력 → ⑥SOP 검색 → ⑦조정값 검증 → ⑧승인 대기 등록"})
    s = _call(log, "get_defect_summary", {})
    if not s["spiking"]:
        text = f"[상황] 급증한 결함 유형 없음 (최근 결함률 {s['total_defect_rate']:.1%}). 조치 불필요."
        log.write("final", {"text": text}); return text
    d = s["spiking"][0]; top = next(x for x in s["by_type"] if x["defect_type"] == d)
    _call(log, "detect_deviation", {})
    _call(log, "get_process_history", {"defect_type": d})
    rc = _call(log, "rank_causes", {"defect_type": d})
    if not rc["candidates"]:
        text = f"[상황] {KO[d]} 급증, 그러나 원인 분석 불가: {rc.get('error')}"
        log.write("final", {"text": text}); return text
    c = rc["candidates"][0]; v = c["var"]
    hist = _call(log, "get_action_history", {"defect_type": d, "var": v})
    sop = _call(log, "search_sop", {"query": f"{KO[d]} {c['desc']} 대응"})
    if not c["adjustable"]:
        text = (f"[상황] {KO[d]} {top['baseline_rate']:.1%} → {top['rate']:.1%}\n"
                f"[원인] {c['station']} · {c['desc']} (SHAP {c['shap_share']:.0%}, {c['z']:+.1f}σ)\n"
                f"[조치 제안] 열연 조정 불가 변수 → 해당 로트 보류 및 제강 공정 통보\n"
                f"[주의] SOP: {sop['results'][0]['section']}")
        log.write("final", {"text": text}); return text
    import yaml
    target = yaml.safe_load(open(ROOT / "config/process_spec.yaml", encoding="utf-8"))["variables"][v]["mean"]
    chk = _call(log, "check_adjustment", {"var": v, "proposed_value": target})
    val = chk["safe_value"]
    if not chk["ok"]:
        chk2 = _call(log, "check_adjustment", {"var": v, "proposed_value": val})
        note = "목표값까지 1회 조정폭 초과 → 1차 조정값만 제안" if chk2["ok"] else "; ".join(chk2["issues"])
    else:
        note = "허용범위·조정폭 검증 통과"
    sub = _call(log, "submit_recommendation", {"defect_type": d, "var": v, "to_value": val,
                                               "reason": f"SHAP {c['shap_share']:.0%}, {c['z']:+.1f}σ 이탈"})
    second = rc["candidates"][1] if len(rc["candidates"]) > 1 else None
    text = (f"[상황] {KO[d]} 발생률 {top['baseline_rate']:.1%} → {top['rate']:.1%} (최근 {s['window']}개)\n"
            f"[원인] 1순위 {c['station']} · {c['desc']}: 최근 평균 {c['recent_mean']}{chk['unit']} ({c['z']:+.1f}σ, {c['direction']}), SHAP 기여 {c['shap_share']:.0%}"
            + (f"\n       2순위 {second['desc']} (SHAP {second['shap_share']:.0%})" if second else "") +
            f"\n[조치 제안] {c['desc']} {chk['current']} → {val}{chk['unit']} ({note})"
            f"\n       SOP 근거: {sop['results'][0]['section']} · 승인 대기 id={sub.get('action_id')}"
            f"\n[주의] 과거 동일 조치 이력 {len(hist['history'])}건. 조정 후 50개 코일 불량률 재확인 필요.")
    log.write("final", {"text": text}); return text


def run(goal: str, offline: bool = False) -> str:
    use_llm = not offline and os.getenv("ANTHROPIC_API_KEY")
    log = RunLog(goal, "claude" if use_llm else "fallback")
    if use_llm:
        try:
            return run_llm(goal, log)
        except Exception as e:
            log.write("error", {"text": f"LLM 호출 실패 → Fallback 전환 ({type(e).__name__}: {e})"})
    return run_fallback(goal, log)
