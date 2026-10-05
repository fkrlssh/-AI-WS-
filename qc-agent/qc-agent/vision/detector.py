"""시연 앱용 결함 검출기
- weights(best.pt)가 있고 ultralytics가 설치돼 있으면 YOLO로 실제 추론
- 없으면 mock 모드: 이미지 옆 라벨(정답)을 그대로 사용 (개발용, 시연·평가 수치에 쓰지 않음)
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Detector:
    def __init__(self, weights: str | None = None, conf: float = 0.25):
        self.conf, self.model, self.mode, self.error = conf, None, "mock", None
        if weights and (ROOT / weights).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(ROOT / weights))
                self.mode = "yolo"
            except Exception as e:  # ultralytics 미설치 등
                self.error = f"{type(e).__name__}: {e}"
        elif weights:
            self.error = f"가중치 파일 없음: {weights}"

    def detect(self, image_path: str) -> list[dict]:
        """반환: [{'cls': 'scratches', 'conf': 0.91, 'xyxyn': [x1, y1, x2, y2]}, ...] (좌표 0~1)"""
        p = ROOT / image_path
        if self.mode == "yolo":
            r = self.model.predict(str(p), conf=self.conf, verbose=False)[0]
            names = self.model.names
            return [{"cls": names[int(c)], "conf": round(float(s), 3), "xyxyn": [round(float(v), 4) for v in b]}
                    for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxyn)]
        lbl = Path(str(p).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        if not lbl.exists():
            return []
        names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
        out = []
        for i, line in enumerate(lbl.read_text().splitlines()):
            v = line.split()
            if len(v) < 5:
                continue
            c, cx, cy, w, h = int(v[0]), *map(float, v[1:5])
            h8 = int(hashlib.md5(f"{image_path}{i}".encode()).hexdigest()[:6], 16) / 0xFFFFFF
            out.append({"cls": names[c], "conf": round(0.80 + 0.18 * h8, 3),
                        "xyxyn": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
        return out
