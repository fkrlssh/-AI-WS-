"""factory.db 접근 헬퍼 — 제품ID 기준으로 공정이력·검사결과를 결합한 wide 테이블 제공"""
import sqlite3
from functools import lru_cache
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = yaml.safe_load(open(ROOT / "config/process_spec.yaml", encoding="utf-8"))
VARS = SPEC["variables"]
STATIONS = SPEC["stations"]
KO = SPEC["defect_names_ko"]

# 시연/평가 시 '현재 시점'을 과거 시점으로 되돌려 재생하기 위한 컨텍스트
CONTEXT = {"now_seq": None}


def db():
    return sqlite3.connect(ROOT / "data/factory.db")


@lru_cache(maxsize=1)
def _wide_all() -> pd.DataFrame:
    con = db()
    prod = pd.read_sql("SELECT * FROM products", con)
    log = pd.read_sql("SELECT product_id, var, value FROM process_log", con)
    ins = pd.read_sql("SELECT product_id, pred_defect, conf, image_path FROM inspections", con)
    wide = log.pivot(index="product_id", columns="var", values="value").reset_index()
    df = prod.merge(wide, on="product_id").merge(ins, on="product_id").sort_values("seq")
    df["pred_defect"] = df["pred_defect"].fillna("normal")
    return df.reset_index(drop=True)


def window(last_n: int, offset: int = 0) -> pd.DataFrame:
    """현재 시점(now_seq)에서 offset만큼 앞선 곳까지의 최근 last_n개 제품"""
    df = _wide_all()
    now = CONTEXT["now_seq"] if CONTEXT["now_seq"] is not None else int(df["seq"].max()) + 1
    end = now - offset
    return df[(df["seq"] >= end - last_n) & (df["seq"] < end)]


def current_value(var: str, last_n: int = 30) -> float:
    return float(window(last_n)[var].mean())
