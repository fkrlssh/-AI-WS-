"""작업표준서(SOP) 검색 — 1차 버전은 TF-IDF(문자 n-gram).
D3에서 임베딩 기반 벡터검색으로 교체 가능 (인터페이스 search(query, k) 유지)."""
import re
from functools import lru_cache
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SOP_DIR = Path(__file__).resolve().parent / "sop"


@lru_cache(maxsize=1)
def _index():
    chunks = []
    for f in sorted(SOP_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        for sec in re.split(r"\n(?=## )", text):
            title = sec.strip().splitlines()[0].lstrip("# ").strip()
            chunks.append({"doc": f.name, "section": title, "text": sec.strip()})
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    mat = vec.fit_transform([c["text"] for c in chunks])
    return chunks, vec, mat


def search(query: str, k: int = 3):
    chunks, vec, mat = _index()
    sims = cosine_similarity(vec.transform([query]), mat)[0]
    top = sims.argsort()[::-1][:k]
    return [{**chunks[i], "score": round(float(sims[i]), 3)} for i in top]
