"""정상 제품에 의사 정상 이미지를 배정 (factory.db inspections.image_path가 비어 있는 제품)

NEU-DET에는 정상 이미지가 없어 정상 제품이 '이미지 없음'으로 처리되던 문제를 보완한다.
의사 정상 이미지(Prepare_normals.py 결과 pn_data/pseudo_normal/{train,test}/*.png)를 data/pseudo_normal/에 복사해 두고 실행.
※ 원본 데이터로 학습한 모델(runs/neu_yolo11n)은 이 이미지를 학습에 쓰지 않았으므로 오검출(헛경보) 측정에도 쓸 수 있다.

  python scripts/assign_normal_images.py
  python vision/infer.py --weights runs/neu_yolo11n/weights/best.pt   # 다시 추론해야 정상 제품의 오검출이 반영됨
"""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PN = ROOT / "data" / "pseudo_normal"

imgs = sorted(p for p in PN.rglob("*.png"))
if not imgs:
    raise SystemExit(f"의사 정상 이미지가 없습니다: {PN} (pn_data/pseudo_normal의 png를 복사하세요)")
con = sqlite3.connect(ROOT / "data/factory.db")
rows = con.execute("SELECT i.product_id, p.seq FROM inspections i JOIN products p USING(product_id) "
                   "WHERE i.image_path IS NULL ORDER BY p.seq").fetchall()
con.executemany("UPDATE inspections SET image_path=? WHERE product_id=?",
                [(imgs[seq % len(imgs)].relative_to(ROOT).as_posix(), pid) for pid, seq in rows])
con.commit()
print(f"[done] 정상 제품 {len(rows)}개에 의사 정상 이미지 {len(imgs)}종 배정")
