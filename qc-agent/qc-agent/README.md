# 비전 AI 기반 불량 분류 및 원인 추적 Agent

제4회 경남 AI·SW 경진대회 (대학부)

## 실행 (Windows PowerShell, 이 폴더에서)
```
python -m pip install -r requirements.txt
copy .env.example .env                       # 로컬 LLM(Ollama) 설정
ollama pull qwen3:8b                         # 로컬 LLM 모델
python scripts/simulate_process.py           # 공정 이력 + 고장 에피소드 → data/factory.db
python scripts/assign_normal_images.py       # 정상 제품에 의사 정상 이미지 배정
python vision/infer.py --weights runs/neu_bg_yolo11n/weights/best.pt
python -m streamlit run app.py               # 시연 앱
```
- Agent 모드(앱 사이드바): 로컬 LLM(Ollama · Qwen3 8B) / 오프라인(규칙·통계 엔진)
- 터미널 조사 모드: `python run_demo.py --scenario scratches --offline` (규칙 기반 플래너, 실행 로그는 `logs/`)

## 사용 모델과 가중치
| 모델 | 버전·위치 | 출처 · 라이선스 | 용도 |
|---|---|---|---|
| YOLO11n (팀 학습, 최종) | `runs/neu_bg_yolo11n/weights/best.pt` — 저장소에 포함 | `yolo11n.pt`(Ultralytics, AGPL-3.0)에서 전이학습, Ultralytics 8.4.172 | 결함 판별 |
| Qwen3 8B | Ollama `qwen3:8b` (ID 500a1f067a9f, 5.2GB) — `ollama pull qwen3:8b` | Alibaba Qwen, Apache-2.0 / Ollama(MIT) | 로컬 LLM: 추가 조사·설명 |

외부 API 모델은 사용하지 않는다. 데이터·라이브러리 출처는 `docs/SOURCES.md`에 있다.

## 평가
| 스크립트 | 내용 | 결과 |
|---|---|---|
| `scripts/eval_vision.py` | 분류 정확도, 추론 시간, 정상 오검출률 | `runs/eval_vision.json` |
| `scripts/eval_cause.py` (`--early N`) | 급증 탐지율, 원인 변수·공정 top-1/top-3 | 화면 출력 |
| `scripts/eval_agent_time.py --provider offline\|ollama` | 조언 표시·최종 답변 시간 | `runs/eval_agent_time_*.json` |
| `scripts/eval_agent_quality.py --provider offline\|ollama` | 답변 문제 유형 자동 집계 | `runs/eval_agent_quality.md/.csv` |

## 구조
```
app.py                     Streamlit 시연 앱 (가상 컨베이어, 검출 보관함, 원인 분석 대화)
config/process_spec.yaml   공정·변수·정상범위·허용한계·결함-원인 가정 (시뮬레이션)
agent/chat.py              제품 단위 대화 Agent (규칙·통계 엔진 + 로컬 LLM 근거 고정형)
agent/tools.py             공통 도구 8종 (+ chat.py의 get_product_trace = 9종)
agent/agent.py             터미널 조사 모드 (규칙 기반 플래너, 실행 로그)
agent/rag.py, agent/sop/   작업표준서(가상) 검색
vision/                    YOLO 학습·추론
scripts/                   데이터 준비, 공정 시뮬레이터, 평가
docs/                      범위 확정서(초기 계획), 출처·AI 활용 기록표
```

> 공정 데이터·작업표준서는 시뮬레이션/가상입니다. 성능 수치는 "시뮬레이션 시나리오 기준"입니다.
> `agent/agent.py`·`agent/chat.py`에 개발 초기에 만든 Claude API 연결 코드가 남아 있으나 사용·검증하지 않았습니다.
