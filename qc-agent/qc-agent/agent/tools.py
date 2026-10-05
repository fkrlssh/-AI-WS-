"""Agent가 호출하는 도구(Tool) 모음.
각 함수는 JSON 직렬화 가능한 dict를 반환 → LLM Agent와 Fallback 플래너가 같은 도구를 공유."""
from datetime import datetime
import numpy as np
import pandas as pd
from .data import CONTEXT, KO, STATIONS, VARS, current_value, db, window
from . import rag

FEATURES = list(VARS.keys())


def _var_info(v):
    s = VARS[v]
    return {"var": v, "desc": s["desc"], "station": STATIONS[s["station"]]["name"], "unit": s["unit"]}


def get_defect_summary(last_n: int = 300, baseline_n: int = 1500) -> dict:
    """최근 last_n개 제품의 결함 유형별 발생률을 직전 baseline_n개와 비교해 급증 유형을 찾는다."""
    cur, base = window(last_n), window(baseline_n, offset=last_n)
    rows = []
    for d in KO:
        c = int((cur.pred_defect == d).sum())
        r, br = c / max(len(cur), 1), float((base.pred_defect == d).mean()) if len(base) else 0.0
        rows.append({"defect_type": d, "name_ko": KO[d], "count": c, "rate": round(r, 4),
                     "baseline_rate": round(br, 4), "lift": round(r / br, 2) if br > 0 else None,
                     "spike": bool(c >= 5 and r >= max(2 * br, 0.01))})
    rows.sort(key=lambda x: (x["spike"], x["count"]), reverse=True)
    total = float((cur.pred_defect != "normal").mean())
    return {"window": last_n, "total_defect_rate": round(total, 4), "by_type": rows,
            "spiking": [r["defect_type"] for r in rows if r["spike"]]}


def detect_deviation(last_n: int = 300) -> dict:
    """최근 구간 공정변수의 정상범위 이탈 여부(평균 z-score, 범위 밖 비율)."""
    w = window(last_n)
    out = []
    for v, s in VARS.items():
        lo, hi = s["normal"]
        z = (w[v].mean() - s["mean"]) / s["std"]
        out.append({**_var_info(v), "mean": round(float(w[v].mean()), 3), "normal": [lo, hi],
                    "z": round(float(z), 2), "out_of_range_ratio": round(float(((w[v] < lo) | (w[v] > hi)).mean()), 3)})
    out.sort(key=lambda x: abs(x["z"]), reverse=True)
    return {"window": last_n, "variables": out, "abnormal": [o["var"] for o in out if o["out_of_range_ratio"] >= 0.2]}


def get_process_history(defect_type: str, last_n: int = 300) -> dict:
    """해당 결함 제품과 정상 제품의 공정변수 평균을 비교(제품ID로 결합된 이력)."""
    w = window(last_n)
    d, n = w[w.pred_defect == defect_type], w[w.pred_defect == "normal"]
    rows = []
    for v, s in VARS.items():
        diff = (d[v].mean() - n[v].mean()) / s["std"] if len(d) and len(n) else 0.0
        rows.append({**_var_info(v), "defect_mean": round(float(d[v].mean()), 3) if len(d) else None,
                     "normal_mean": round(float(n[v].mean()), 3), "gap_std": round(float(diff), 2)})
    rows.sort(key=lambda x: abs(x["gap_std"]), reverse=True)
    sample = d.tail(3)[["product_id", "ts", "image_path"]].to_dict("records")
    return {"defect_type": defect_type, "n_defect": int(len(d)), "n_normal": int(len(n)),
            "variables": rows[:5], "recent_samples": sample}


def rank_causes(defect_type: str, last_n: int = 300, train_n: int = 2000) -> dict:
    """설명가능 AI(SHAP)로 원인 공정·변수 후보를 순위화.
    최근 train_n개 제품으로 '결함 발생 여부' 분류 모델을 학습하고, 최근 last_n 구간 결함 제품의 SHAP 기여도를 집계."""
    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    tr = window(train_n)
    y = (tr.pred_defect == defect_type).astype(int)
    if y.sum() < 10:
        return {"defect_type": defect_type, "error": "해당 결함 표본이 10건 미만이라 원인 분석 신뢰도가 낮음", "candidates": []}
    Xs = (tr[FEATURES] - [VARS[v]["mean"] for v in FEATURES]) / [VARS[v]["std"] for v in FEATURES]
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(Xs, y)
    recent = window(last_n)
    target = recent[recent.pred_defect == defect_type]
    if len(target) == 0:
        return {"defect_type": defect_type, "error": "최근 구간에 해당 결함 없음", "candidates": []}
    Xt = (target[FEATURES] - [VARS[v]["mean"] for v in FEATURES]) / [VARS[v]["std"] for v in FEATURES]
    sv = shap.TreeExplainer(model).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else sv
    contrib = np.clip(sv, 0, None).mean(axis=0)          # 결함 확률을 '올린' 기여만 집계
    total = contrib.sum() or 1.0
    cands = []
    for i in np.argsort(contrib)[::-1][:5]:
        v = FEATURES[i]; zt = float(Xt[v].mean())
        cands.append({**_var_info(v), "shap_share": round(float(contrib[i] / total), 3),
                      "recent_mean": round(float(target[v].mean()), 3), "z": round(zt, 2),
                      "direction": "high" if zt > 0 else "low", "adjustable": VARS[v].get("adjustable", True)})
    return {"defect_type": defect_type, "name_ko": KO.get(defect_type), "n_explained": int(len(target)),
            "model_train_size": int(len(tr)), "candidates": cands}


def search_sop(query: str, k: int = 3) -> dict:
    """작업표준서(SOP)에서 관련 조치 기준을 검색(RAG)."""
    return {"query": query, "results": rag.search(query, k)}


def check_adjustment(var: str, proposed_value: float) -> dict:
    """조정 제안값이 설비 허용 한계·1회 최대 조정폭·정상범위를 지키는지 검증하고, 위반 시 안전한 값을 제시."""
    if var not in VARS:
        return {"ok": False, "issues": [f"알 수 없는 변수: {var}"]}
    s = VARS[var]
    if not s.get("adjustable", True):
        return {"ok": False, "var": var, "issues": ["열연 공정에서 조정 불가(상류 공정 변수) → 제강 공정 통보 필요"]}
    cur = current_value(var)
    lo_h, hi_h = s["hard"]; lo_n, hi_n = s["normal"]; step = s["max_step"]
    issues, safe = [], float(proposed_value)
    if not lo_h <= safe <= hi_h:
        issues.append(f"허용 한계 {lo_h}~{hi_h}{s['unit']} 벗어남"); safe = min(max(safe, lo_h), hi_h)
    if step and abs(safe - cur) > step:
        issues.append(f"1회 최대 조정폭 {step}{s['unit']} 초과 (현재 {cur:.2f})")
        safe = cur + np.sign(safe - cur) * step
    if not lo_n <= safe <= hi_n:
        issues.append(f"조정 후 값이 정상범위 {lo_n}~{hi_n} 밖 → 추가 단계 조정 필요")
    return {"var": var, "current": round(cur, 3), "proposed": proposed_value, "ok": not issues,
            "issues": issues, "safe_value": round(float(safe), 3), "unit": s["unit"],
            "normal": [lo_n, hi_n], "hard": [lo_h, hi_h], "max_step": step}


def get_action_history(defect_type: str | None = None, var: str | None = None, limit: int = 5) -> dict:
    """과거 조치(승인·거절·효과) 이력 조회 — Agent의 장기 기억(Memory)."""
    q, p = "SELECT * FROM actions WHERE 1=1", []
    if defect_type: q += " AND defect_type=?"; p.append(defect_type)
    if var: q += " AND var=?"; p.append(var)
    q += " ORDER BY action_id DESC LIMIT ?"; p.append(limit)
    con = db(); con.row_factory = __import__("sqlite3").Row
    return {"history": [dict(r) for r in con.execute(q, p).fetchall()]}


def submit_recommendation(defect_type: str, var: str, to_value: float, reason: str) -> dict:
    """최종 조정 제안을 '승인 대기(pending)'로 등록. 실제 설비 적용은 작업자 승인 후에만 이뤄진다."""
    chk = check_adjustment(var, to_value)
    if not chk.get("ok"):
        return {"registered": False, "reason": "검증 실패 — check_adjustment 결과의 safe_value로 다시 제안하세요", "check": chk}
    rate = float((window(300).pred_defect == defect_type).mean())
    con = db()
    cur = con.execute("INSERT INTO actions(ts, defect_type, var, from_value, to_value, status, reason, rate_before) "
                      "VALUES (?,?,?,?,?,?,?,?)", (datetime.now().isoformat(timespec="seconds"), defect_type, var,
                                                   chk["current"], to_value, "pending", reason, round(rate, 4)))
    con.commit()
    return {"registered": True, "action_id": cur.lastrowid, "status": "pending(작업자 승인 대기)", "check": chk}


TOOLS = [get_defect_summary, detect_deviation, get_process_history, rank_causes,
         search_sop, check_adjustment, get_action_history, submit_recommendation]
