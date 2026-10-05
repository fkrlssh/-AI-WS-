"""검출 이미지(제품) 단위 대화형 Agent
- 사용자가 보관함에서 검출 이미지를 고르면, 그 제품ID를 기준으로 공정 이력·원인·조치를 묻고 답한다.
- LLM 모드: Claude(LangChain) tool calling. 대화 이력 유지.
- 오프라인 모드: 같은 도구를 규칙대로 호출해 답한다(API 키가 없거나 호출 실패 시).
"""
import json, os, sqlite3
from . import tools as T
from .agent import _brief
from .data import CONTEXT, KO, SPEC, STATIONS, VARS, db

MAX_STEPS = 10


# ---------------------------------------------------------------- 제품 단위 도구
def get_product_trace(product_id: str) -> dict:
    """선택한 제품(코일)이 각 공정을 지날 때의 시각과 공정변수 값을 정상범위와 비교한다(제품ID 기준 역추적)."""
    con = db()
    p = con.execute("SELECT seq, ts, lot FROM products WHERE product_id=?", (product_id,)).fetchone()
    if not p:
        return {"error": f"제품 {product_id} 없음"}
    det = con.execute("SELECT defect, conf FROM detections WHERE product_id=?", (product_id,)).fetchone() \
        if con.execute("SELECT name FROM sqlite_master WHERE name='detections'").fetchone() else None
    rows = con.execute("SELECT station, var, value, ts FROM process_log WHERE product_id=?", (product_id,)).fetchall()
    trace = []
    for station, var, value, ts in rows:
        s = VARS[var]; lo, hi = s["normal"]
        z = (value - s["mean"]) / s["std"]
        trace.append({"station": STATIONS[station]["name"], "station_order": STATIONS[station]["order"],
                      "var": var, "desc": s["desc"], "value": round(value, 3), "unit": s["unit"],
                      "normal": [lo, hi], "z": round(z, 2), "out_of_range": not lo <= value <= hi, "passed_at": ts})
    trace.sort(key=lambda r: (r["station_order"], r["var"]))
    by_var = {r["var"]: r for r in trace}
    known = []
    if det:   # 작업표준서·공정사양의 결함-원인 매핑 후보 + 이 제품 통과 시 값(원인 방향으로 얼마나 치우쳤는지)
        for c in SPEC["defect_causes"].get(det[0], []):
            r = by_var[c["var"]]
            known.append({"var": c["var"], "desc": r["desc"], "station": r["station"], "direction": c["direction"],
                          "value": r["value"], "unit": r["unit"], "normal": r["normal"],
                          "z_toward_cause": round(r["z"] if c["direction"] == "high" else -r["z"], 2),
                          "adjustable": VARS[c["var"]].get("adjustable", True)})
        known.sort(key=lambda k: k["z_toward_cause"], reverse=True)
    return {"product_id": product_id, "seq": p[0], "inspected_at": p[1], "lot": p[2],
            "defect": det[0] if det else None, "defect_ko": KO.get(det[0]) if det else None,
            "conf": det[1] if det else None, "trace": trace, "known_causes": known,
            "out_of_range": [f"{r['station']}·{r['desc']} {r['value']}{r['unit']} (정상 {r['normal'][0]}~{r['normal'][1]})"
                             for r in trace if r["out_of_range"]]}


CHAT_TOOLS = [get_product_trace] + T.TOOLS

SYSTEM = """당신은 열연 강판 공장의 품질 원인추적 AI Agent입니다. 작업자가 검출 이미지 하나를 골라 질문합니다.
선택된 제품: {pid} / 검출 결함: {defect_ko}({defect}) / 신뢰도 {conf} / 검사 시각 {ts} / LOT {lot}

규칙
1. 먼저 get_product_trace로 이 제품이 각 공정을 지날 때의 값을 확인하세요(제품 단위 근거).
2. 원인은 rank_causes(최근 300개 제품의 SHAP 순위)와 이 제품의 이탈 값을 함께 근거로 대세요. 둘을 구분해서 말하세요.
   rank_causes가 표본 부족으로 실패하면 get_product_trace의 known_causes(작업표준서 매핑 + 이 제품 값)로 추정하고, 단발성일 수 있다고 밝히세요.
3. 숫자는 도구 결과에 있는 값만 쓰세요. 추측으로 만들지 마세요.
4. adjustable=false인 변수는 조정하지 말고 상류(제강) 통보를 제안하세요.
5. 조치를 물으면 search_sop 근거 + check_adjustment 검증을 거친 값만 제안하세요(실패 시 safe_value).
6. submit_recommendation(승인 대기 등록)은 사용자가 등록·승인 요청을 명시했을 때만 호출하세요.
7. 이 제품의 공정변수가 모두 정상범위이고 원인 후보도 |z|<2이면 공정 기인으로 단정하지 말고, 단발성 결함 또는 오검출(특히 신뢰도 0.5 미만) 가능성을 말하고 조정 대신 관찰·육안 재검사를 권고하세요.
8. 짧고 분명하게, 한국어로, 작업자가 바로 이해할 수 있게 답하세요. 핵심 수치는 굵게 표시해도 됩니다."""


def _call(name, args, steps):
    fn = {f.__name__: f for f in CHAT_TOOLS}[name]
    try:
        r = fn(**args)
    except Exception as e:
        r = {"error": f"{type(e).__name__}: {e}"}
    brief = _brief(name, r) if name != "get_product_trace" else \
        f"이탈 {len(r.get('out_of_range', []))}건: " + ", ".join(r.get("out_of_range", [])[:3])
    steps.append({"name": name, "args": args, "brief": brief})
    return r


def _set_time(pid):
    seq = db().execute("SELECT seq FROM products WHERE product_id=?", (pid,)).fetchone()
    CONTEXT["now_seq"] = (seq[0] + 1) if seq else None   # '그 제품이 검출된 시점'의 데이터만 보도록


# ---------------------------------------------------------------- LLM 모드
def _chat_llm(pid, history, question, steps):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import StructuredTool

    tr = get_product_trace(pid)
    llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"), temperature=0, max_tokens=1500) \
        .bind_tools([StructuredTool.from_function(f) for f in CHAT_TOOLS])
    msgs = [SystemMessage(SYSTEM.format(pid=pid, defect=tr.get("defect"), defect_ko=tr.get("defect_ko"),
                                        conf=tr.get("conf"), ts=tr.get("inspected_at"), lot=tr.get("lot")))]
    for h in history:
        msgs.append(HumanMessage(h["text"]) if h["role"] == "user" else AIMessage(h["text"]))
    msgs.append(HumanMessage(question))
    for _ in range(MAX_STEPS):
        ai = llm.invoke(msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            return ai.text if isinstance(getattr(ai, "text", None), str) else str(ai.content)
        for tc in ai.tool_calls:
            r = _call(tc["name"], tc["args"], steps)
            msgs.append(ToolMessage(json.dumps(r, ensure_ascii=False, default=str), tool_call_id=tc["id"]))
    return "조사 단계가 너무 길어져 중단했습니다. 질문을 나눠서 다시 물어봐 주세요."


# ---------------------------------------------------------------- 오프라인(규칙) 모드
def _chat_offline(pid, question, steps):
    q = question
    tr = _call("get_product_trace", {"product_id": pid}, steps)
    d = tr.get("defect")
    if not d:
        return "이 제품은 결함이 검출되지 않았습니다."
    want_cause = any(k in q for k in ("공정", "원인", "어디", "왜", "발생", "분석"))
    want_hist = any(k in q for k in ("이력", "과거", "전에", "기록"))
    want_action = any(k in q for k in ("조치", "조정", "어떻게", "해결", "권고", "제안")) and not want_hist
    want_reg = any(k in q for k in ("등록", "승인", "올려"))
    if not (want_cause or want_action or want_hist or want_reg):
        want_cause = True

    lines = [f"**[제품]** {pid} · {tr['inspected_at'][11:19]} · LOT {tr['lot']} · **{tr['defect_ko']}** (신뢰도 {tr['conf']})"]
    rc = _call("rank_causes", {"defect_type": d}, steps)
    top = rc["candidates"][0] if rc.get("candidates") else None
    mine = {r["var"]: r for r in tr["trace"]}
    rule_based = False
    if top is None and tr["known_causes"]:   # 같은 결함 표본이 적으면 SHAP 대신 매핑 규칙 + 이 제품 값으로 추정
        k = tr["known_causes"][0]
        top = {**k, "shap_share": None}
        rule_based = True

    calm = top and not mine[top["var"]]["out_of_range"] and abs(top.get("z", 0) or 0) < 2 and abs(mine[top["var"]]["z"]) < 2
    if want_cause:
        if tr["out_of_range"]:
            lines.append("**[이 제품의 공정 이력]** 정상범위를 벗어난 값: " + "; ".join(tr["out_of_range"]))
        else:
            lines.append("**[이 제품의 공정 이력]** 정상범위를 벗어난 공정변수 없음")
        if top and rule_based:
            m = mine[top["var"]]
            lines.append(f"**[원인 추정 · 규칙 기반]** 최근 같은 결함 표본이 10건 미만이라 통계(SHAP) 대신 작업표준서의 "
                         f"결함-원인 매핑과 이 제품의 값으로 추정합니다 → {top['station']} 공정의 **{top['desc']}**, "
                         f"이 제품 통과 시 **{m['value']}{m['unit']}** (정상 {m['normal'][0]}~{m['normal'][1]}, {m['z']:+.1f}σ). "
                         "단발성 불량일 수 있으니 같은 결함이 더 쌓이는지 지켜봐야 합니다.")
        elif top:
            m = mine[top["var"]]
            lines.append(f"**[원인 추정]** {top['station']} 공정의 **{top['desc']}** — 이 제품 통과 시 "
                         f"**{m['value']}{m['unit']}** (정상 {m['normal'][0]}~{m['normal'][1]}, {m['z']:+.1f}σ), "
                         f"최근 {rc['n_explained']}건의 같은 결함에서 SHAP 기여 **{top['shap_share']:.0%}**")
            if len(rc["candidates"]) > 1:
                c2 = rc["candidates"][1]
                lines.append(f"2순위 후보: {c2['station']} · {c2['desc']} (SHAP {c2['shap_share']:.0%})")
        else:
            lines.append(f"**[원인 추정]** 분석 불가: {rc.get('error')}")
        if calm and not tr["out_of_range"]:
            low = tr["conf"] is not None and tr["conf"] < 0.5
            lines.append("**[판단]** 이 제품의 공정변수와 원인 후보 값이 모두 정상범위 안이라 공정에서 생긴 불량으로 보기 어렵습니다. "
                         + ("검출 신뢰도도 낮아(0.5 미만) 오검출 가능성이 있으니 육안 재검사를 권고합니다."
                            if low else "단발성 결함으로 보고, 같은 결함이 이어지는지 지켜봅니다."))

    if (want_action or want_reg) and calm:
        lines.append(f"**[조치]** 원인 후보인 {top['desc']}가 이 제품 통과 시와 최근 구간 모두 정상범위 안이라, 단발성 불량으로 보고 "
                     "지금은 조정하지 않고 관찰을 권고합니다. 같은 결함이 이어지면 다시 분석합니다. (승인 대기 등록 안 함)")
    elif (want_action or want_reg) and top:
        if not top["adjustable"]:
            lines.append("**[조치]** 열연 공정에서 조정할 수 없는 상류 변수입니다 → 해당 LOT 보류, 제강 공정 통보")
        else:
            sop = _call("search_sop", {"query": f"{KO[d]} {top['desc']} 대응"}, steps)
            chk = _call("check_adjustment", {"var": top["var"], "proposed_value": VARS[top["var"]]["mean"]}, steps)
            val = chk["safe_value"]
            if not chk["ok"]:
                chk = _call("check_adjustment", {"var": top["var"], "proposed_value": val}, steps)
            note = "1회 조정폭 제한으로 1차 조정값" if val != VARS[top["var"]]["mean"] else "목표값"
            lines.append(f"**[조치 제안]** {top['desc']} **{chk['current']} → {val}{chk['unit']}** ({note}, 허용범위 검증 통과)")
            sec = sop["results"][0]
            bullet = next((l.strip("- ").strip() for l in sec["text"].splitlines() if top["desc"][:4] in l or "정상범위" in l), "")
            lines.append(f"SOP 근거: 「{sec['section']}」 {bullet}")
            if want_reg:
                why = "규칙 기반 추정" if rule_based else f"SHAP {top['shap_share']:.0%}"
                sub = _call("submit_recommendation", {"defect_type": d, "var": top["var"], "to_value": val,
                                                      "reason": f"{pid} 검출 기반, {why}"}, steps)
                lines.append(f"**[등록]** 승인 대기 id={sub.get('action_id')} — 작업자 승인 후 적용됩니다."
                             if sub.get("registered") else f"등록 실패: {sub.get('reason')}")

    if want_hist:
        h = _call("get_action_history", {"defect_type": d}, steps)["history"]
        lines.append("**[과거 조치]** " + ("없음" if not h else
                     "; ".join(f"#{x['action_id']} {x['var']} {x['from_value']}→{x['to_value']} ({x['status']})" for x in h[:3])))
    return "\n\n".join(lines)


def chat(pid: str, history: list, question: str, offline: bool = False) -> dict:
    """반환: {'text': 답변, 'steps': [도구 호출 기록], 'mode': 'claude'|'offline'}"""
    _set_time(pid)
    steps = []
    use_llm = not offline and os.getenv("ANTHROPIC_API_KEY")
    if use_llm:
        try:
            return {"text": _chat_llm(pid, history, question, steps), "steps": steps, "mode": "claude"}
        except Exception as e:
            steps.append({"name": "fallback", "args": {}, "brief": f"LLM 실패 → 오프라인 전환 ({type(e).__name__})"})
    return {"text": _chat_offline(pid, question, steps), "steps": steps, "mode": "offline"}
