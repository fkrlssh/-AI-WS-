"""품질 원인추적 Agent 시연 앱
  streamlit run app.py

① 가상 컨베이어: 제품 이미지가 흘러가고 검사 카메라 위치에서 YOLO가 판정(불량은 박스·결함명 표시)
② 검출 보관함: 불량으로 판정된 이미지가 자동 저장·누적 (detections/ 폴더 + factory.db detections 테이블)
③ 원인 분석 대화: 보관함 이미지를 고르면 그 제품ID로 Agent와 대화 (어느 공정에서 발생했는지, 조치, 이력, 승인 대기 등록)
"""
import base64, io, json, os, sqlite3, time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
from agent.data import KO  # noqa: E402
from agent import chat as C  # noqa: E402
from vision.detector import Detector  # noqa: E402

st.set_page_config(page_title="품질 원인추적 Agent", layout="wide")
DB = ROOT / "data" / "factory.db"
SAVE_DIR = ROOT / "detections"
SAVE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 화면 상수
W, H = 1180, 300               # 컨베이어 캔버스
S, GAP = 150, 46               # 제품 크기, 간격
P = S + GAP
BELT_Y = 70
CAM_X = 760                    # 검사 카메라 위치(제품 왼쪽 끝 기준)
RED, GREEN, BLUE = (220, 45, 45), (34, 150, 80), (30, 120, 230)


def font(size, bold=False):
    for f in (["malgunbd.ttf", "NotoSansCJK-Bold.ttc", "AppleSDGothicNeo.ttc"] if bold else
              ["malgun.ttf", "NotoSansCJK-Regular.ttc", "AppleSDGothicNeo.ttc"]):
        for base in ("C:/Windows/Fonts/", "/usr/share/fonts/opentype/noto/", "/System/Library/Fonts/", ""):
            try:
                return ImageFont.truetype(base + f, size)
            except OSError:
                continue
    return ImageFont.load_default()


F_SM, F_MD, F_LB = font(13, True), font(15), font(14, True)


def con():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS detections(product_id TEXT PRIMARY KEY, seq INTEGER, ts TEXT, lot TEXT,
                 defect TEXT, conf REAL, boxes_json TEXT, image_path TEXT, saved_path TEXT, latency_ms REAL, detected_at TEXT)""")
    return c


# ---------------------------------------------------------------- 데이터
@st.cache_data
def load_products():
    c = con()
    df = pd.read_sql("""SELECT p.product_id, p.seq, p.ts, p.lot, i.image_path, i.gt_defect
                        FROM products p JOIN inspections i USING(product_id) ORDER BY p.seq""", c)
    pn = sorted((ROOT / "data" / "pseudo_normal").rglob("*.png"))
    if pn:  # 정상 제품 이미지가 비어 있으면 의사 정상 이미지로 채워 보여줌
        miss = df.image_path.isna()
        df.loc[miss, "image_path"] = [pn[s % len(pn)].relative_to(ROOT).as_posix() for s in df.loc[miss, "seq"]]
    return df


@st.cache_data
def load_episodes():
    return pd.read_sql("SELECT * FROM fault_episodes ORDER BY episode_id", con())


@st.cache_resource
def get_detector(weights, conf):
    return Detector(weights or None, conf=conf)


@lru_cache(maxsize=512)
def tile(path):
    try:
        return Image.open(ROOT / path).convert("RGB").resize((S, S))
    except Exception:
        return Image.new("RGB", (S, S), (120, 120, 120))


def annotate(img, boxes, size):
    im = img.convert("RGB").resize((size, size))
    d = ImageDraw.Draw(im)
    for b in boxes:
        x1, y1, x2, y2 = (int(v * size) for v in b["xyxyn"])
        d.rectangle([x1, y1, x2, y2], outline=RED, width=max(2, size // 100))
        if size >= 300:
            d.text((x1 + 3, y1 + 2), f"{KO.get(b['cls'], b['cls'])} {b['conf']:.2f}", font=F_LB, fill=RED)
    return im


# ---------------------------------------------------------------- 상태
def init_state(start_seq):
    st.session_state.update(start_seq=start_seq, cursor=float(CAM_X - 3 * P), judged=0, results={},
                            running=False, selected=None)


if "cursor" not in st.session_state:
    init_state(int(load_episodes().iloc[0].start_seq) - 30)
st.session_state.setdefault("chats", {})


def judge(idx, det):
    """제품 idx(스트림 내 순번)를 판정하고, 불량이면 보관함에 저장"""
    prods = load_products()
    r = prods.iloc[st.session_state.start_seq + idx]
    t0 = time.perf_counter()
    boxes = det.detect(r.image_path)
    ms = (time.perf_counter() - t0) * 1000
    defect = max(boxes, key=lambda b: b["conf"])["cls"] if boxes else None
    st.session_state.results[idx] = {"boxes": boxes, "defect": defect, "ms": ms}
    if defect:
        out = SAVE_DIR / f"{r.product_id}_{defect}.jpg"
        annotate(Image.open(ROOT / r.image_path), boxes, 400).save(out, quality=92)
        c = con()
        c.execute("INSERT OR REPLACE INTO detections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (r.product_id, int(r.seq), r.ts, r.lot, defect, max(b["conf"] for b in boxes),
                   json.dumps(boxes), r.image_path, out.relative_to(ROOT).as_posix(), round(ms, 1),
                   datetime.now().isoformat(timespec="seconds")))
        c.commit()


def render_belt(offset):
    prods = load_products()
    im = Image.new("RGB", (W, H), (244, 246, 249))
    d = ImageDraw.Draw(im)
    d.rectangle([0, BELT_Y - 16, W, BELT_Y + S + 16], fill=(66, 69, 76))
    for x in range(-int(offset) % 36 - 36, W, 36):
        d.line([x, BELT_Y - 16, x + 18, BELT_Y + S + 16], fill=(82, 85, 93), width=3)
    cur = st.session_state.cursor
    k_lo, k_hi = int(np.ceil((cur - W) / P)), int(np.floor((cur + S) / P))
    for k in range(max(0, k_lo), k_hi + 1):
        if st.session_state.start_seq + k >= len(prods):
            continue
        x = int(cur - k * P)
        r = prods.iloc[st.session_state.start_seq + k]
        res = st.session_state.results.get(k)
        img = annotate(tile(r.image_path), res["boxes"], S) if res else tile(r.image_path)
        im.paste(img, (x, BELT_Y))
        if res:
            ng = res["defect"] is not None
            d.rectangle([x, BELT_Y + S + 1, x + S, BELT_Y + S + 20], fill=RED if ng else GREEN)
            d.text((x + 5, BELT_Y + S + 2), f"불량 · {KO[res['defect']]}" if ng else "양품", font=F_SM, fill="white")
    d.rectangle([CAM_X - 8, BELT_Y - 26, CAM_X + S + 8, BELT_Y + S + 26], outline=BLUE, width=3)
    d.text((CAM_X - 6, BELT_Y - 50), "검사 카메라 (YOLO11 판정)", font=F_MD, fill=BLUE)
    d.text((14, 12), f"→ 진행 방향   현재 제품 {prods.iloc[min(len(prods) - 1, st.session_state.start_seq + st.session_state.judged)].product_id}",
           font=F_MD, fill=(60, 60, 60))
    return im


def advance(px, det):
    st.session_state.cursor += px
    while st.session_state.cursor - st.session_state.judged * P >= CAM_X:
        judge(st.session_state.judged, det)
        st.session_state.judged += 1


# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("설정")
    eps = load_episodes()
    labels = {f"ep{r.episode_id} {KO[r.defect_type]} 급증 (seq {r.start_seq}~)": int(r.start_seq) for r in eps.itertuples()}
    choice = st.selectbox("시나리오(고장 에피소드)", list(labels))
    weights = st.text_input("YOLO 가중치", "runs/neu_bg_yolo11n/weights/best.pt")
    conf_th = st.slider("불량 판정 신뢰도 기준", 0.25, 0.90, 0.25, 0.05,
                        help="이 값 이상인 박스만 불량으로 판정. 높이면 오검출↓ 미검출↑")
    det = get_detector(weights, conf_th)
    st.caption(f"검출 모드: **{det.mode}**" + (f" — {det.error}" if det.error else ""))
    speed = st.slider("벨트 속도", 4, 40, 14)
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    offline = st.toggle("오프라인 Agent(규칙 기반)", value=not has_key, disabled=not has_key)
    c1, c2 = st.columns(2)
    if c1.button("처음부터", width="stretch"):
        init_state(labels[choice] - 30); st.rerun()
    if c2.button("보관함 비우기", width="stretch"):
        c = con(); c.execute("DELETE FROM detections"); c.commit()
        for f in SAVE_DIR.glob("*.jpg"):
            f.unlink()
        st.session_state.selected = None; st.rerun()
    st.divider()
    st.subheader("승인 대기 조치")
    pend = con().execute("SELECT action_id, defect_type, var, from_value, to_value FROM actions WHERE status='pending' "
                         "ORDER BY action_id DESC").fetchall()
    if not pend:
        st.caption("없음")
    for aid, dft, var, fv, tv in pend:
        st.write(f"#{aid} {KO[dft]} · {var} {fv:.2f}→{tv:.2f}")
        a1, a2 = st.columns(2)
        if a1.button("승인", key=f"ap{aid}"):
            c = con(); c.execute("UPDATE actions SET status='approved', note=? WHERE action_id=?", ("대시보드 승인", aid)); c.commit(); st.rerun()
        if a2.button("거절", key=f"rj{aid}"):
            c = con(); c.execute("UPDATE actions SET status='rejected', note=? WHERE action_id=?", ("대시보드 거절", aid)); c.commit(); st.rerun()

# ---------------------------------------------------------------- 상단: 컨베이어
st.title("비전 AI 기반 불량 분류 · 원인 추적 Agent")
b1, b2, b3, _ = st.columns([1, 1, 1.3, 5])
if b1.button("▶ 시작" if not st.session_state.running else "⏸ 정지", width="stretch"):
    st.session_state.running = not st.session_state.running; st.rerun()
if b2.button("⏩ 20개", width="stretch", help="제품 20개를 빠르게 검사"):
    advance(20 * P, det); st.rerun()
if b3.button("⏭ 이상 구간으로", width="stretch", help="시나리오 고장 시작 지점까지 빠르게 검사"):
    target = labels[choice] - st.session_state.start_seq + 25
    if target > st.session_state.judged:
        advance((target - st.session_state.judged) * P, det)
    st.rerun()


@st.fragment(run_every=0.15 if st.session_state.running else None)
def conveyor():
    before = st.session_state.judged
    if st.session_state.running:
        advance(speed, det)
    # st.image는 빠른 자동 갱신 중 미디어 파일이 만료돼 깨진 이미지가 뜰 수 있어 data URI로 직접 넣는다
    buf = io.BytesIO()
    render_belt(st.session_state.cursor).save(buf, "JPEG", quality=82)
    st.markdown(f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}" '
                'style="width:100%;border-radius:6px;display:block">', unsafe_allow_html=True)
    res = [st.session_state.results[k] for k in sorted(st.session_state.results)]
    ng = [r for r in res if r["defect"]]
    recent = res[-50:]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("검사 수", len(res))
    k2.metric("양품", len(res) - len(ng))
    k3.metric("불량", len(ng))
    k4.metric("최근 50개 불량률", f"{sum(r['defect'] is not None for r in recent) / max(1, len(recent)):.0%}")
    k5.metric("평균 판정 시간", f"{np.mean([r['ms'] for r in res]):.1f} ms" if res else "-")
    if st.session_state.judged != before and any(st.session_state.results[k]["defect"] for k in range(before, st.session_state.judged)):
        st.rerun(scope="app")   # 새 불량 → 보관함 갱신


conveyor()

# ---------------------------------------------------------------- 하단: 보관함 + 대화
left, right = st.columns([1.05, 1.35], gap="large")
with left:
    dets = pd.read_sql("SELECT * FROM detections ORDER BY seq DESC", con())
    dets = dets[[bool(x) and (ROOT / x).is_file() for x in dets.saved_path]]   # 파일이 지워진 기록은 숨김
    st.subheader(f"검출 이미지 보관함 ({len(dets)})")
    if dets.empty:
        st.info("불량으로 판정된 이미지가 여기에 모입니다.")
    flt = st.multiselect("결함 필터", sorted(dets.defect.unique()) if len(dets) else [], format_func=lambda x: KO[x])
    if flt:
        dets = dets[dets.defect.isin(flt)]
    cols = st.columns(4)
    for i, r in enumerate(dets.head(40).itertuples()):
        with cols[i % 4]:
            st.image(str(ROOT / r.saved_path), width="stretch")
            st.caption(f"{KO[r.defect]} · {r.conf:.2f}\n{r.product_id}")
            if st.button("원인 분석", key=f"sel_{r.product_id}", width="stretch",
                         type="primary" if st.session_state.selected == r.product_id else "secondary"):
                st.session_state.selected = r.product_id
                st.session_state.running = False
                st.rerun()

with right:
    pid = st.session_state.selected
    if not pid:
        st.subheader("원인 분석 대화")
        st.info("왼쪽 보관함에서 이미지를 고르면, 그 제품이 어느 공정에서 문제가 생겼는지 Agent와 대화할 수 있습니다.")
    else:
        r = pd.read_sql("SELECT * FROM detections WHERE product_id=?", con(), params=(pid,)).iloc[0]
        st.subheader(f"원인 분석 대화 · {pid}")
        a, b = st.columns([1, 1.6])
        a.image(str(ROOT / r.saved_path), width="stretch")
        tr = C.get_product_trace(pid)
        b.markdown(f"**결함** {KO[r.defect]} (신뢰도 {r.conf:.2f})  \n**검사 시각** {r.ts[11:19]} · **LOT** {r.lot}  \n"
                   f"**판정 시간** {r.latency_ms:.1f} ms  \n**정상범위 이탈** " + (", ".join(tr["out_of_range"]) or "없음"))
        tdf = pd.DataFrame([{"공정": t["station"], "변수": t["desc"], "통과 시 값": f"{t['value']:g} {t['unit']}",
                             "정상범위": f"{t['normal'][0]:g}~{t['normal'][1]:g}", "이탈": "⚠" if t["out_of_range"] else ""}
                            for t in tr["trace"]])
        with st.expander("이 제품의 공정 이력 (제품ID 기준 역추적)", expanded=False):
            st.dataframe(tdf.style.apply(lambda s: ["background-color:#fde2e2" if s["이탈"] else "" for _ in s], axis=1),
                         hide_index=True, width="stretch")

        hist = st.session_state.chats.setdefault(pid, [])
        box = st.container(height=420)
        with box:
            for m in hist:
                with st.chat_message(m["role"]):
                    st.markdown(m["text"])
                    if m.get("steps"):
                        with st.expander(f"Agent 조사 과정 ({len(m['steps'])}단계 · {m.get('mode')})"):
                            for s_ in m["steps"]:
                                st.markdown(f"🔧 `{s_['name']}` → {s_['brief']}")
        sug = ["이 결함은 어느 공정에서 발생했어?", "어떻게 조치해야 해?", "과거에 비슷한 조치가 있었어?", "조치안을 승인 대기로 등록해줘"]
        sc = st.columns(len(sug))
        q = None
        for i, s_ in enumerate(sug):
            if sc[i].button(s_, key=f"sug{i}", width="stretch"):
                q = s_
        q = st.chat_input("이 제품에 대해 질문하세요 (예: 같은 LOT에서 또 나왔어?)") or q
        if q:
            with st.spinner("Agent가 공정 이력을 조사하는 중..."):
                ans = C.chat(pid, [{"role": m["role"], "text": m["text"]} for m in hist], q, offline=offline)
            hist += [{"role": "user", "text": q},
                     {"role": "assistant", "text": ans["text"], "steps": ans["steps"], "mode": ans["mode"]}]
            st.rerun()
