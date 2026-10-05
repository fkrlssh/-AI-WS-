# 출처·AI 활용 기록표 (별지2 작성용 — 개발하면서 계속 추가)

| 구분 | 이름 | 출처 | 라이선스/조건 | 사용 위치 | 기존자산/신규 |
|---|---|---|---|---|---|
| 데이터셋 | NEU Surface Defect Database (NEU-DET) | Northeastern University, Song Kechen et al. / YOLO 변환 미러: github.com/Marfbin/NEU-DET-with-yolov8 | 학술 공개 데이터 (라이선스 원문 확인 필요) | 비전 학습·검사 스트림 | 외부 데이터 |
| 모델 | YOLO11n 사전학습 가중치 | Ultralytics | AGPL-3.0 | 결함 검출 | 기존자산(전이학습) |
| 라이브러리 | ultralytics, scikit-learn, shap, pandas, langchain-core, langchain-anthropic | PyPI | 각 라이선스 | 전반 | 외부 |
| LLM API | Claude (Anthropic) | api.anthropic.com | 이용약관 | Agent 판단 | 외부 API |
| AI 코딩도구 | Claude | claude.ai | — | D1~2 코드 골격·문서 초안 작성 보조 | 신고 대상 |
| 시뮬레이션 데이터 | 열연 공정변수 이력 | 팀 자체 생성 (`scripts/simulate_process.py`) | — | 원인 추적 | 신규 (가정값 명시) |
| 문서 | 열연 품질 작업표준서 | 팀 작성 가상 문서 | — | RAG | 신규 (가상 문서 명시) |

## 신규개발분 (대회 기간 9/29~)
- 공정 시뮬레이터, 원인 추적 도구(SHAP), Agent 도구 8종, Agent 루프·Fallback, 평가 스크립트, SOP 문서
