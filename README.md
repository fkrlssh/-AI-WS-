# 비전 AI 기반 제조 공정 불량품 분류 및 원인 추적 Agent

제4회 경남 AI·SW 경진대회 출품작. 컨베이어를 지나는 열연 강판의 표면 결함을 YOLO11로 판별하고, 제품 ID로 공정 이력을 역추적해 원인 공정과 조정값을 제안하는 AI Agent.

## 폴더 구성
| 경로 | 내용 |
| --- | --- |
| `qc-agent/qc-agent/` | 본 프로젝트 (Streamlit 시연 앱, Agent, 비전, 공정 시뮬레이션, 평가 스크립트) — 실행법은 해당 폴더 README |
| `Prepare_normals.py` | NEU-DET에서 의사 정상(무결함) 이미지 생성 |
| `augment_data.py` | 데이터 증강 실험 (최종 모델에는 미채택) |
| `train_yolo.py` | 초기 학습 스크립트 |
| `데이터_구성_의사정상_증강.md` | 데이터 구성 설명 |

## 최종 모델
`qc-agent/qc-agent/runs/neu_bg_yolo11n/weights/best.pt` — NEU-DET + 의사 정상 배경 이미지 281장으로 학습한 YOLO11n

| 지표 | 값 |
| --- | --- |
| mAP@0.5 (val 180장) | 0.776 |
| 이미지 분류 정확도 | 95.0% |
| 정상 오검출률 (의사 정상 test 36장) | 16.7% |
| 원인 변수 top-1 (시뮬레이션 12개 에피소드, 종료 시점 / 발생 후 40개) | 100% / 92% |
| 1장 판별 시간 (RTX 3060) | 9.2ms |
| 검증된 조언 표시 시간 | 오프라인 30건 평균 0.56초(최대 0.73초) · 로컬 LLM 3회 36건 모두 1초 이내(최대 0.70초) |
| 로컬 LLM(Qwen3 8B) 설명 완료 시간 | 같은 코드로 3회 측정(각 12건), 회차별 평균 11.5~21.1초, 최대 27.1초 — `qc-agent/qc-agent/runs/eval_agent_time_runs.md` |
| 로컬 LLM 설명 품질 (답변 12건 사람 검토) | 10건에 표현 오류, 최종 수정 전 버전으로 생성한 답변 — `qc-agent/qc-agent/runs/eval_agent_quality_review.md` |

## 실행
```
cd qc-agent/qc-agent
pip install -r requirements.txt
python scripts/simulate_process.py
python scripts/assign_normal_images.py
python vision/infer.py --weights runs/neu_bg_yolo11n/weights/best.pt
python -m streamlit run app.py
```
- Agent 모드: 로컬 LLM(Ollama · Qwen3 8B, `.env`에 `LLM_PROVIDER=ollama`) · 오프라인(규칙·통계 엔진)
- 평가: `scripts/eval_vision.py`, `eval_cause.py`, `eval_agent_time.py`, `eval_agent_quality.py` (`eval_cause.py`는 화면 출력, 나머지는 `runs/eval_*`)

데이터셋 이미지(NEU-DET)와 `.env`는 저장소에 포함하지 않는다. 공정 데이터는 모두 시뮬레이션이다.
