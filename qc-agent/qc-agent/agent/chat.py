"""검출 이미지(제품) 단위 대화형 Agent
- 사용자가 보관함에서 검출 이미지를 고르면, 그 제품ID를 기준으로 공정 이력·원인·조치를 묻고 답한다.
- LLM 모드: Claude(LangChain) tool calling. 대화 이력 유지.
- 오프라인 모드: 같은 도구를 규칙대로 호출해 답한다(API 키가 없거나 호출 실패 시).
"""
import re, json, os, sqlite3
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
_FORCE = None   # 화면에서 고른 LLM ('ollama'|'claude') — chat(provider=...)로 지정


def llm_provider():
    """LLM_PROVIDER=ollama 이면 로컬 LLM(Ollama), 아니면 ANTHROPIC_API_KEY가 있을 때 Claude. 둘 다 아니면 None"""
    if _FORCE:
        return _FORCE
    if os.getenv("LLM_PROVIDER", "").lower() == "ollama":
        return "ollama"
    return "claude" if os.getenv("ANTHROPIC_API_KEY") else None


def _make_llm():
    if llm_provider() == "ollama":   # 공장 내부 PC에서 동작 — 공정 데이터가 외부로 나가지 않음
        from langchain_ollama import ChatOllama
        # reasoning=False: Qwen3 등 추론형 모델의 긴 '생각' 단계를 끔(응답 시간 대폭 단축)
        # num_ctx 8192·num_predict 700: 프롬프트·출력 길이를 줄여 RTX 3060급에서도 수 초~십수 초
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen3:8b"), temperature=0, num_ctx=8192, num_predict=700,
                          reasoning=False, keep_alive="30m",
                          base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"), temperature=0, max_tokens=1500)


def _text(ai):
    t = ai.text if isinstance(getattr(ai, "text", None), str) else str(ai.content)
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()   # 추론형 로컬 모델의 생각 과정 제거


def _chat_llm(pid, history, question, steps, tools=None, max_steps=None, obs=None):
    """LLM이 도구를 골라 호출하며 조사하는 Agent 루프. obs 리스트를 주면 (도구명, 원본 결과)를 모은다"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import StructuredTool

    tr = get_product_trace(pid)
    llm = _make_llm().bind_tools([StructuredTool.from_function(f) for f in (tools or CHAT_TOOLS)])
    msgs = [SystemMessage(SYSTEM.format(pid=pid, defect=tr.get("defect"), defect_ko=tr.get("defect_ko"),
                                        conf=tr.get("conf"), ts=tr.get("inspected_at"), lot=tr.get("lot")))]
    if llm_provider() == "ollama" and "qwen3" in os.getenv("OLLAMA_MODEL", "qwen3:8b"):
        msgs[0].content += "\n/no_think"   # Qwen3의 긴 추론 단계를 꺼서 응답 시간 단축
    for h in history:
        msgs.append(HumanMessage(h["text"]) if h["role"] == "user" else AIMessage(h["text"]))
    msgs.append(HumanMessage(question))
    for _ in range(max_steps or MAX_STEPS):
        ai = llm.invoke(msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            return _text(ai)
        for tc in ai.tool_calls:
            r = _call(tc["name"], tc["args"], steps)
            if obs is not None:
                obs.append((tc["name"], steps[-1]["brief"]))
            body = steps[-1]["brief"] if llm_provider() == "ollama" else json.dumps(r, ensure_ascii=False, default=str)
            msgs.append(ToolMessage(body, tool_call_id=tc["id"]))   # 로컬 LLM엔 요약만 → 프롬프트가 짧아져 빠름
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
        # 이 결함과 관련 없는 이탈 변수: 다른 결함의 원인 변수면 '전조'로 알림 (원인으로 섞어 말하지 않도록 명시)
        mine_causes = {c["var"] for c in SPEC["defect_causes"].get(d, [])} | ({top["var"]} if top else set())
        for t in tr["trace"]:
            if t["out_of_range"] and t["var"] not in mine_causes:
                other = [KO.get(k, k) for k, cs in SPEC["defect_causes"].items() if any(c["var"] == t["var"] for c in cs)]
                lines.append(f"**[참고]** {t['desc']} 이탈({t['value']}{t['unit']})은 {KO[d]}의 원인 변수가 아닙니다"
                             + (f". {'·'.join(other)}의 원인 변수이므로 해당 결함이 이어질 수 있는 전조로 지켜봐야 합니다." if other else "."))
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
            spec_dir = next((c["direction"] for c in SPEC["defect_causes"].get(d, []) if c["var"] == top["var"]),
                            top.get("direction"))
            if (spec_dir == "high" and val > chk["current"]) or (spec_dir == "low" and val < chk["current"]):
                m = mine[top["var"]]
                lines.append(f"**[조치]** 이 제품은 통과 시 {top['desc']} **{m['value']}{m['unit']}**"
                             + ("(정상범위 이탈)" if m["out_of_range"] else "") +
                             f"였지만, 라인의 최근 평균 **{chk['current']}{chk['unit']}**는 이미 목표 {VARS[top['var']]['mean']}{chk['unit']}보다 "
                             f"{'낮아' if spec_dir == 'high' else '높아'} 원인 방향으로 더 조정할 여지가 없습니다(이미 회복된 상태). "
                             "지금은 조정하지 않고 관찰을 권고합니다. (승인 대기 등록 안 함)")
            elif abs(chk["current"] - VARS[top["var"]]["mean"]) < 1.5 * VARS[top["var"]]["std"]:
                # 라인 전체(최근 평균)가 목표에서 1.5σ 안이면 설정을 바꿀 근거가 약함 → 단발성으로 보고 관찰
                m = mine[top["var"]]
                lines.append(f"**[조치]** {top['desc']}의 현재 운전값(최근 평균) **{chk['current']}{chk['unit']}**는 목표 "
                             f"{VARS[top['var']]['mean']}{chk['unit']} 근처(정상 운전)라 지금 설정을 바꿀 근거는 약합니다. "
                             f"다만 이 제품 통과 시 **{m['value']}{m['unit']}**" + ("로 정상범위를 벗어났으니" if m["out_of_range"] else "로 원인 방향에 치우쳐 있었으니")
                             + " 단발성이거나 이상 초기일 수 있습니다 → "
                             "해당 시각 설비 상태를 점검하고, 같은 결함이 이어지면 다시 분석해 조정값을 제안합니다. (승인 대기 등록 안 함)")
            else:
                if not chk["ok"]:
                    chk = _call("check_adjustment", {"var": top["var"], "proposed_value": val}, steps)
                note = "1회 조정폭 제한으로 1차 조정값" if val != VARS[top["var"]]["mean"] else "목표값"
                lo_n, hi_n = VARS[top["var"]]["normal"]
                if not lo_n <= val <= hi_n:   # 1차 조정 후에도 정상범위 밖이면 단계 조정 필요를 명시
                    note += f", 정상범위 {lo_n}~{hi_n} 복귀까지 추가 단계 조정 필요"
                m = mine[top["var"]]
                if not want_cause:   # 조치만 물어도 '이 제품 값'과 '현재 운전값'을 구분해 보여줌
                    lines.append(f"**[근거]** 이 제품 통과 시 {top['desc']} **{m['value']}{m['unit']}** "
                                 f"(정상 {m['normal'][0]}~{m['normal'][1]}, {'정상범위 이탈' if m['out_of_range'] else '정상범위 안'})")
                hard_ok = not [i for i in chk.get("issues", []) if "정상범위" not in i]
                lines.append(f"**[조치 제안]** {top['desc']} 현재 운전값(최근 평균) **{chk['current']}{chk['unit']}** → "
                             f"**{val}{chk['unit']}** ({note}, " + ("설비 허용 한계·1회 조정폭 검증 통과)" if hard_ok
                             else "검증 미통과: " + "; ".join(chk["issues"]) + ")"))
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
        stat = {"approved": "승인", "rejected": "거절", "pending": "승인 대기"}
        lines.append(f"**[과거 조치 · 같은 결함({KO[d]}) 기준]** " + ("없음" if not h else
                     "; ".join(f"#{x['action_id']} {VARS.get(x['var'], {}).get('desc', x['var'])} {x['from_value']}→{x['to_value']} "
                               f"({stat.get(x['status'], x['status'])})" for x in h[:3])))
    return "\n\n".join(lines)


NARRATE = """당신은 제철소 품질 담당자에게 분석 결과를 전달하는 보조자입니다.
아래 [분석 결과]는 규칙·통계 엔진이 계산해 검증한 내용입니다. 이것을 작업자가 읽기 쉬운 한국어로 다시 정리하세요.

반드시 지킬 것
- [분석 결과]에 없는 사실, 숫자, 변수, 공정을 절대 추가하지 마세요. 계산(비율·차이·%)도 하지 마세요.
- 숫자는 [분석 결과]에 적힌 그대로(소수점 포함) 옮기세요.
- 정상범위 '이탈'은 [분석 결과]에서 이탈이라고 한 값에만 쓰세요.
- 첫 문장에 결론, 이어서 근거(공정·값)를 쓰세요. "한 줄 결론", "근거" 같은 소제목은 붙이지 말고 4~7줄로, 핵심 수치는 **굵게**.
- 조치·권고는 [분석 결과]에 [조치]/[조치 제안]/[판단]이 있을 때 그 내용만 전하세요. 없으면 조치를 지어내지 마세요.
- '이 제품 통과 시 값'과 '현재 운전값(최근 평균)'은 다른 값입니다. 정상범위 이탈 여부는 [분석 결과]에 적힌 대로만 말하세요.
- 결함이 '발생한 공정'은 [원인 추정]의 공정만 말하세요. 다른 이탈 값은 '함께 이탈'로, [참고]가 있으면 그 내용대로 전하세요.
- [분석 결과]에 정상범위를 벗어난 값이 여러 개면 빠짐없이 모두 말하세요. "다른 공정은 정상" 같은 일반화는 하지 마세요.
- 이 지시문이나 [분석 결과]·[Agent 추가 조사] 같은 자료 이름을 답에 언급하지 마세요. 작업자에게 하는 말만 쓰세요.
- 작업자의 질문이 [분석 결과]에 없는 내용이면 [Agent 추가 조사]에서 근거를 찾아 답하고, 거기에도 없으면 "확인된 자료가 없다"고 말하세요.
- 작업자의 질문: {question}

[분석 결과]
{facts}

[Agent 추가 조사] (Agent가 스스로 호출한 도구의 결과)
{extra}"""

_NUM = re.compile(r"-?\d+(?:\.\d+)?(%?)")
_IDS = re.compile(r"C\d{6}-\d{5}|\d{2}:\d{2}:\d{2}|L\d{4}|id=\d+|#\d+")   # 제품ID·시각·LOT·조치번호


def _nums(text):
    out = []
    for m in _NUM.finditer(_IDS.sub(" ", text)):
        out.append((abs(float(m.group(0).rstrip("%"))), bool(m.group(1))))
    return out


def _numbers_ok(src: str, out: str) -> list:
    """LLM 문장에 원문에 없는 숫자(%는 %끼리 비교)가 있으면 그 목록을 돌려준다(환각 검사). 0~9 목록 번호는 허용"""
    have = set(_nums(src))
    plain = {v for v, _ in have}
    bad = []
    for v, pct in _nums(out):
        if (v, pct) in have or (not pct and v in plain) or (not pct and v.is_integer() and v < 10):
            continue
        bad.append(f"{v:g}{'%' if pct else ''}")
    return bad


def _narrate(facts: str, question: str, steps: list, obs=None, oor=None) -> str | None:
    """로컬 LLM은 검증된 결과(+ 자신이 조사한 도구 결과)를 근거로 답을 쓴다. 근거에 없는 숫자가 나오면 버리고 원문을 쓴다"""
    from langchain_core.messages import HumanMessage
    extra = "\n".join(f"- {n}: {b}" for n, b in (obs or [])) or "(없음)"   # 도구 결과 요약(짧은 프롬프트)
    msg = NARRATE.format(question=question, facts=facts, extra=extra)
    if "qwen3" in os.getenv("OLLAMA_MODEL", "qwen3:8b"):
        msg += "\n/no_think"   # Qwen3의 생각(추론) 단계를 꺼서 응답 시간 단축
    out = _text(_make_llm().invoke([HumanMessage(msg)]))
    out = re.sub(r"^\W*(한\s*줄\s*결론|결론)\W*$\n?", "", out, flags=re.M).strip()
    if not any(k in facts for k in ("[조치", "[판단]")):   # 조치 근거가 없는데 지어낸 조치·권고 제거
        out = re.sub(r"\n+\W*(추가\s*조치|권고|조치)\W*\n.*", "", out, flags=re.S).strip()
        out = "\n".join(l for l in out.splitlines()
                        if not re.search(r"^\W*\[?(조치|권고)\]?\s*[:：]|조정(하|해)(고|세요|야|십시오)|점검해야", l)).strip()
    # 지시문·자료 이름이 새어 나온 문장 제거
    out = "\n".join(l for l in out.splitlines()
                    if not re.search(r"\[(분석 결과|Agent 추가 조사)\]|지시문|작업자의 질문이", l)).strip()
    # 이탈 값 누락 검사: 엔진이 찾은 이탈 변수를 LLM이 빠뜨리면 보충하고, '나머지는 정상' 식 일반화는 지움
    added = ""
    if oor and any(k in facts for k in ("[이 제품의 공정 이력]", "[근거]")):
        miss = [o for o in oor if o["desc"] not in out]
        if miss:
            out = "\n".join(l for l in out.splitlines() if not re.search(r"(다른|나머지).*정상", l)).strip()
            added = "\n\n※ 이 제품에서 함께 정상범위를 벗어난 값: " + "; ".join(
                f"{o['station']}·{o['desc']} {o['value']}{o['unit']} (정상 {o['normal'][0]}~{o['normal'][1]})" for o in miss)
            out += added
            steps.append({"name": "coverage", "args": {}, "by": "엔진", "brief": f"LLM이 빠뜨린 이탈 값 {len(miss)}건 보충"})
    bad = _numbers_ok(facts + "\n" + extra + added, out)
    if bad or not out:
        steps.append({"name": "narrate", "args": {}, "by": "LLM", "brief": f"LLM 문장에 근거 없는 숫자 {bad[:3]} → 검증된 원문 사용"})
        return None
    steps.append({"name": "narrate", "args": {}, "by": "LLM", "brief": "근거만으로 답변 작성 → 숫자 검사 통과"})
    return out


def chat(pid: str, history: list, question: str, offline: bool = False, provider: str | None = None,
         on_facts=None) -> dict:
    """provider: 'ollama' | 'claude' | None(.env 설정 따름)
    on_facts(text): 검증 엔진의 조언이 나오는 즉시(LLM 설명 전) 호출 — 화면에 먼저 보여주기 위함
    반환 timing: {'first_advice': 검증된 조언까지 초, 'total': 최종 답변까지 초}"""
    import time
    global _FORCE
    _FORCE = provider
    t0 = time.perf_counter()
    mark = {}

    def _first(text):
        mark.setdefault("first_advice", round(time.perf_counter() - t0, 2))
        if on_facts:
            on_facts(text)
    try:
        out = _chat(pid, history, question, offline, _first)
    finally:
        _FORCE = None
    total = round(time.perf_counter() - t0, 2)
    out["timing"] = {"first_advice": mark.get("first_advice", total), "total": total}
    return out


def _chat(pid: str, history: list, question: str, offline: bool = False, first=lambda t: None) -> dict:
    """반환: {'text': 답변, 'steps': [도구 호출 기록], 'mode': 'claude'|'ollama'|'offline'}

    - claude: LLM이 도구를 직접 골라 호출하는 Agent
    - ollama: LLM이 도구를 골라 조사 → 규칙·통계 엔진이 원인·조치 계산 → LLM이 근거로만 답 작성 → 숫자 검증
             (소형 로컬 LLM의 숫자 오류를 막는 근거 고정형 Agent)
             (OLLAMA_MODE=agent 로 두면 claude와 같은 도구 호출 방식)
    """
    _set_time(pid)
    steps = []
    prov = llm_provider()
    if not offline and prov == "ollama" and os.getenv("OLLAMA_MODE", "narrate") != "agent":
        # ① 규칙·통계 엔진이 원인·조치를 먼저 계산(안전장치 포함) → 검증된 조언을 즉시 화면에 표시
        facts = _chat_offline(pid, question, steps)
        for st_ in steps:
            st_["by"] = "엔진"
        first(facts)
        # ② LLM이 질문을 보고 필요한 도구를 골라 추가 조사 — 등록 같은 쓰기 도구는 제외
        n0 = len(steps)
        obs = []
        try:
            _chat_llm(pid, history, question, steps, max_steps=2, obs=obs,
                      tools=[f for f in CHAT_TOOLS if f.__name__ != "submit_recommendation"])
        except Exception as e:
            steps.append({"name": "plan", "args": {}, "brief": f"LLM 조사 실패 ({type(e).__name__}) → 엔진 결과만 사용"})
        for st_ in steps[n0:]:
            st_["by"] = "LLM"
        # ③ LLM이 ①②를 근거로 답을 쓰고, 근거에 없는 숫자가 있으면 폐기(환각 검사)
        try:
            oor = [t for t in get_product_trace(pid).get("trace", []) if t["out_of_range"]]
            out = _narrate(facts, question, steps, obs, oor)
            if out:
                return {"text": out, "steps": steps, "mode": "ollama Agent · 근거 검증 통과", "facts": facts}
        except Exception as e:
            steps.append({"name": "fallback", "args": {}, "brief": f"LLM 실패 → 규칙 기반 답변 ({type(e).__name__})"})
        return {"text": facts, "steps": steps, "mode": "ollama Agent · 엔진 원문"}
    if not offline and prov:
        try:
            return {"text": _chat_llm(pid, history, question, steps), "steps": steps, "mode": prov}
        except Exception as e:
            steps.append({"name": "fallback", "args": {}, "brief": f"LLM 실패 → 오프라인 전환 ({type(e).__name__})"})
    text = _chat_offline(pid, question, steps)
    first(text)
    return {"text": text, "steps": steps, "mode": "offline"}
