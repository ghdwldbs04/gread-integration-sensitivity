# Numerical Integration Sensitivity in Graph Neural Reaction–Diffusion Networks

KSC 2026 학부생 논문 실험 코드. 반응항(Blurring-Sharpening, Allen-Cahn, 순수 확산) × 수치 적분 설정(Euler/RK4 step size, dopri5 tolerance)이 GREAD의 정확도·발산·NFE·Dirichlet energy에 미치는 영향을 측정한다.

## 구성
- `rd_sweep.py` — 단일 설정 실행기 (best params 적용 후 적분 설정을 덮어씀, split별 시드 고정, NaN/NFE 한도/솔버 오류를 발산으로 기록)
- `run_sweep.sh` — 전체 스윕 (재실행 시 완료된 설정은 건너뜀)
- `analyze.py` — 결과 JSONL → 표(Markdown/CSV) + 그림
- `results/` — 실험 결과

## 실행
GREAD 원본 코드(https://github.com/jeongwhanchoi/GREAD, Apache-2.0)를 clone하고, geom-gcn split(https://github.com/graphdml-uiuc-jlu/geom-gcn)을 `GREAD/src/splits/`에 복사한 뒤, 이 저장소의 세 파일을 `GREAD/src/`에 링크해서 실행한다.
환경: Python 3.9, torch 2.7.1+cu128, torch-geometric 2.3.1, torchdiffeq 0.2.3 (RTX 5060 Ti, sm_120)

## 참고
Choi et al., GREAD: Graph Neural Reaction-Diffusion Networks, ICML 2023.
