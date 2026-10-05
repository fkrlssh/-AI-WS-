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
| 원인 변수 top-1 (시뮬레이션 12개 에피소드, 종료 시점) | 100% |

데이터셋 이미지(NEU-DET)와 API 키(.env)는 저장소에 포함하지 않는다.
