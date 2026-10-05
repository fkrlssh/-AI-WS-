# 출처·AI 활용 기록표 (별지2 작성용)

| 구분 | 이름 | 출처 | 라이선스/조건 | 사용 위치 | 기존자산/신규 |
|---|---|---|---|---|---|
| 데이터셋 | NEU Surface Defect Database (NEU-DET) | Northeastern University, Song Kechen et al. / YOLO 변환 미러: github.com/Marfbin/NEU-DET-with-yolov8 | 학술 공개 데이터 (라이선스 원문 확인 필요) | 비전 학습·검사 스트림 | 외부 데이터 |
| 모델 | YOLO11n 사전학습 가중치 | Ultralytics | AGPL-3.0 | 결함 검출 | 기존자산(전이학습) |
| 로컬 LLM | Qwen3 8B (`qwen3:8b`) | Alibaba Qwen, Ollama로 실행 | Apache-2.0 (모델), MIT (Ollama) | Agent 설명 작성·추가 조사 | 외부 모델 |
| 라이브러리 | ultralytics, opencv-python, scikit-learn, shap, pandas, numpy, langchain-core, langchain-ollama, streamlit | PyPI | 각 라이선스 | 전반 | 외부 |
| AI 코딩도구 | Claude | claude.ai | — | 코드 작성·디버깅, 평가 스크립트, 문서 초안 작성 보조 | 신고 대상 |
| 시뮬레이션 데이터 | 열연 공정변수 이력 | 팀 자체 생성 (`scripts/simulate_process.py`) | — | 원인 추적 | 신규 (가정값 명시) |
| 합성 데이터 | 의사 정상 이미지 317장 | 팀 자체 생성 (`Prepare_normals.py`, NEU-DET에서 결함을 지워 합성) | NEU-DET 조건 따름 | 비전 학습 배경·오검출 평가·시연 | 신규 (합성 명시) |
| 문서 | 열연 품질 작업표준서 | 팀 작성 가상 문서 | — | SOP 검색 | 신규 (가상 문서 명시) |

## 사용 모델과 가중치 (모델명 · 버전 · 출처 · 라이선스 · 활용 범위)

| 모델 | 버전 | 출처 | 라이선스 | 활용 범위 | 가중치 제공 |
|---|---|---|---|---|---|
| YOLO11n 사전학습 가중치 | `yolo11n.pt` (Ultralytics 8.4.170·8.4.172로 학습) | Ultralytics (github.com/ultralytics/ultralytics) | AGPL-3.0 | 결함 판별 모델의 전이학습 시작 가중치 | 공개 사전학습 모델 — 별도 제출 없음 |
| 팀 학습 YOLO11n (최종) | `runs/neu_bg_yolo11n/weights/best.pt` (2026-10-04 학습, 5.5MB) | 팀 학습: NEU-DET 1,620장 + 합성 정상 배경 281장, 100에폭 | 학습 결과물(기반 모델 AGPL-3.0 준수) | 시연 앱·추론·평가의 결함 판별 | **저장소에 포함** (실행·검증용) |
| Qwen3 8B | Ollama `qwen3:8b` (모델 ID 500a1f067a9f, 5.2GB) | Alibaba Qwen / ollama.com/library/qwen3 | Apache-2.0 | 로컬 LLM: 추가 조사 도구 선택과 설명 작성 (원인·조치 판단은 규칙·통계 엔진) | 공개 사전학습 모델 — `ollama pull qwen3:8b` |
| Ollama | 로컬 LLM 실행기 | ollama.com | MIT | Qwen3 8B 로컬 실행 | — |

- 외부 API 모델은 사용하지 않았다. (Claude API 연결 코드가 `agent/agent.py`·`agent/chat.py`에 남아 있으나 사용·검증하지 않음)
- 초기·증강 실험 모델(`neu_yolo11n`, `neu_aug_yolo11n`)은 비교 실험용이며 최종 결과에 쓰지 않았다.

## 신규개발분 (대회 기간 9/29~)
- 공정 시뮬레이터, 의사 정상 이미지 생성, 원인 추적 도구(SHAP), Agent 도구 9종, 근거 고정형 대화 Agent(로컬 LLM)·터미널 조사 모드·Fallback, 시연 앱, 평가 스크립트, SOP 문서

> `langchain-anthropic`은 requirements에 남아 있으나 사용하지 않았다.
