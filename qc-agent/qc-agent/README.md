# Promptors · 비전 AI 기반 불량 분류 및 원인 추적 Agent

제4회 경남 AI·SW 경진대회 (대학부) — D1~2 골격

## 빠른 시작
```bash
pip install -r requirements.txt
python scripts/prepare_data.py          # NEU-DET 다운로드 → data/neu-det
python scripts/simulate_process.py      # 공정 이력 + 고장 에피소드 → data/factory.db
python vision/infer.py --mock           # (모델 학습 전) 정답 라벨로 검사결과 채우기
python scripts/eval_cause.py            # 원인 추적 정확도 (시뮬레이션 기준)
python run_demo.py --scenario scratches --offline   # Fallback 플래너로 시연
cp .env.example .env  # 키 입력 후
python run_demo.py --scenario scratches              # Claude Agent로 시연
python run_demo.py approve 1                          # 작업자 승인 → 다음 실행 때 Memory로 참조
```

## YOLO 학습 (GPU: Colab 또는 경남TP GPU 서버)
```bash
# Colab
!git clone <repo> && cd qc-agent && pip install -r requirements.txt
!python scripts/prepare_data.py
!python vision/train.py --model yolo11n.pt --epochs 100
# 학습 후 best.pt를 받아와서
python vision/infer.py --weights runs/neu_yolo11n/weights/best.pt
```

## 구조
```
config/process_spec.yaml   공정·변수·정상범위·허용한계·결함-원인 가정 (시뮬레이션)
scripts/                   데이터 준비, 공정 시뮬레이터, 원인추적 평가
vision/                    YOLO 학습·추론 (--mock 지원)
agent/tools.py             Agent 도구 8종
agent/agent.py             Claude(LangChain) tool-calling 루프 + Fallback 플래너 + 실행 로그
agent/sop/                 RAG용 작업표준서(가상)
docs/SCOPE.md              D1 범위 확정서 / docs/SOURCES.md 출처 기록표
logs/                      실행 로그(jsonl) — 대시보드에서 판단 과정 표시용
```

> ⚠️ 공정 데이터·작업표준서는 시뮬레이션/가상입니다. 성능 수치는 "시뮬레이션 시나리오 기준"으로만 보고합니다.
> ⚠️ `--mock` 결과는 개발용이며 시연·평가에 사용하지 않습니다. `.env`(API 키)는 절대 커밋하지 않습니다.
