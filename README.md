# 기후변화 하 미국 옥수수 수확량 예측과 작물 배분 최적화

미국 카운티 단위 기후·토양 자료로 옥수수·대두 수확량을 예측하고, 그 예측치를 계수로 삼아
**+2℃ 온난화 시나리오 하의 작물 배분을 조합 최적화(GA)로** 푸는 프로젝트입니다.

> **핵심 메시지 — "더위가 옥수수를 해치고, 적은 재배치로 손실을 상당 부분 만회할 수 있다."**

머신러닝 모델 2개(수확량 예측 · 기후 시나리오) + 최적화(GA)로 구성됩니다.
데이터는 미국 카운티 단위 농업·기후 패널(1981–2015), 2,644개 카운티 · 70,721개 관측입니다.

---

## 주요 결과

| 항목 | 결과 |
|---|---|
| 옥수수 수확량 예측 | LightGBM, **CV R² 0.732** (RMSE 20.08 bu/ac) |
| 대두 수확량 예측 | LightGBM, **CV R² 0.807** (RMSE 4.62 bu/ac) |
| 고온 피해 (고정효과) | 29℃ 초과 노출 대체효과 **−1.67 bu/ac/일** (t = −60.7) |
| +2℃ 무조정 영향 | 옥수수 수확량 **−8.98%**, 카운티의 79.2%가 손해 |
| GA 재배치 | 농지 **11.2%만 바꿔 손실의 16.0% 회복** (238 / 2,142 카운티) |
| Greedy 대비 | Greedy는 **2.7배 많은 면적**(29.9%)을 갈아엎고 22.6% 회복 |
| 트레이드오프 스윗스팟 | 무릎 **λ = 42 $/ac** |

**핵심 통찰 — 평균기온이 아니라 온도 노출 분포가 문제다.** 노출가중 평균기온이 사실상 동일한
(20.51℃ vs 20.52℃) 두 카운티·연도가, 29℃ 초과 노출일수는 44.8일 대 7.3일로 갈리고
수확량은 95.4 대 153.0 bu/ac로 갈린다.

전체 분석은 **[`outputs/corn_project_report.docx`](outputs/corn_project_report.docx)** (최종 보고서)
와 **[`project_proposal_v3.md`](project_proposal_v3.md)** (기획·정식화)를 참조하세요.

---

## 파이프라인

노트북은 **번호 순서대로 실행**해야 합니다. 각 단계가 앞 단계의 산출물(`data/processed/`)에 의존합니다.

| # | 노트북 | 하는 일 | 주요 산출물 |
|---|---|---|---|
| 01 | `01_preprocessing.ipynb` | 6개 원본을 `stco`(+`year`)로 병합, 결측 처리 | `corn_panel.parquet` (70,721 × 136) |
| 02 | `02_eda.ipynb` | 29℃ 임계 신호, 다중공선성, 매칭 사례 | fig01–fig09 |
| 03 | `03_features.ipynb` | 피처 엔지니어링 (전략 A: 도메인 압축 / B: 원자료+정규화) | `features_stratA/B.parquet`, `feature_columns.json` |
| 04 | `04_yield_model.ipynb` | 옥수수 9개 모델 × 2전략 비교 + 고정효과 해석 모델 | `yield_model_final.joblib`, fig10–fig16 |
| 04b | `04b_soybean_model.ipynb` | 대두 모델 (동일 파이프라인) | `soybean_model_final.joblib`, fig23–fig26 |
| 05 | `05_climate_model.ipynb` | 기후 추세 + **+2℃ 시나리오 생성** (184일 제약 보존) | `scenarios.parquet`, fig17–fig22 |
| 06 | `06_optimization.ipynb` | **단작 조합 최적화 (GA)** + 전환비용 트레이드오프 | `optimization_results.parquet`, fig27–fig31 |

> **주의:** `04b`는 `04`의 피처 정의에, `05`는 `04`의 모델에, `06`은 `04`·`04b`·`05` 모두에 의존합니다.

### 방법론 요지

- **평가는 `GroupKFold(stco)` 5-fold.** 무작위 분할은 같은 카운티의 다른 연도가 학습·평가에 동시에
  들어가 정보가 샙니다(RF 기준 R² 0.654 → 0.448). 전 과정에서 카운티 단위 분할을 씁니다.
- **184일 제약.** 온도 노출 121구간의 행별 합은 항상 184일(생육기간)입니다. 따라서 개별 계수가 아니라
  **대체효과**(29℃ 이하 1일 → 29℃ 초과 1일)로만 해석해야 합니다.
- **목적함수는 순이익 $/ac.** 옥수수 부셸과 대두 부셸은 더할 수 없는 단위입니다. bu/ac를 그대로 쓰면
  2,142개 카운티 전부에서 옥수수가 이겨 최적화가 퇴화합니다 (`06` 부록에 실증).

---

## 실행법

### 1. 환경

Python 3.11 기준입니다.

```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. 데이터 내려받기 (별도 필요)

**원본 데이터는 이 저장소에 포함되어 있지 않습니다** (약 350MB, 재배포 라이선스 문제).
아래 6종을 `data/raw/` 에 직접 넣어야 합니다.

```
data/raw/
├── yielddata.csv          # 카운티×연도 옥수수·대두·면화·밀 수확량 (bu/ac)
├── gddMarAug.csv          # 온도 노출 분포 121구간 (3–8월). 행 합 = 184일
├── gddAprOct.csv          # 온도 노출 분포 (4–10월). 본 분석은 MarAug 사용
├── pptMarAug.csv          # 생육기간 강수량 (mm)
├── pptAprOct.csv
├── soil1992.csv           # 토양 특성 (보수력·사토·점토·유기물·pH 등)
├── soil2001.csv
├── soil2006.csv
├── soil2011.csv
├── gridInfo.csv           # 카운티 내 농지 규모 (numAg) — 최적화의 A_c
└── cntymap/               # 카운티 경계 셰이프파일 (코로플레스 지도용)
    ├── cntymap.shp
    ├── cntymap.shx
    └── cntymap.dbf
```

이 자료는 Schlenker & Roberts (2009, PNAS)의 **Agricultural Climate Data Collection (ACDC)** 계열
데이터셋입니다. 자료 정의는 `data/raw/Manual_ACDCv1_20170814.pdf` 를 참조하세요.
공통 키는 `stco`(5자리 FIPS = 주 2자리 + 카운티 3자리)이고, 시계열 파일은 `year`(1981–2015)를 추가 키로 갖습니다.

> `data/raw/` 는 **읽기 전용**입니다. 모든 파생 산출물은 `data/processed/` 에 씁니다.

### 3. 노트북 실행

```bash
jupyter lab            # 01 → 02 → 03 → 04 → 04b → 05 → 06 순서로
```

전체를 한 번에 돌리려면:

```bash
cd notebooks
for nb in 01_preprocessing 02_eda 03_features 04_yield_model 04b_soybean_model 05_climate_model 06_optimization; do
  py -3.11 -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 "$nb.ipynb"
done
```

> `04`(트리 모델 18개 조합)와 `06`(GA λ 스윕)이 가장 오래 걸립니다. 각각 30분 내외를 예상하세요.
> GA는 `seed=42`로 고정되어 재현 가능합니다.

---

## 저장소 구조

```
corn-project/
├── notebooks/            # 01~06 분석 노트북 (실행 결과·그림 포함)
├── src/
│   └── optimize_ga.py
├── outputs/
│   ├── figures/          # fig01 ~ fig31 (31개)
│   ├── results/          # 모델 비교·고정효과·λ 스윕 등 표 (CSV)
│   └── corn_project_report.docx   # 최종 보고서
├── data/                 # ⚠ 저장소에 미포함 — 직접 준비
│   ├── raw/              #    원본 (읽기 전용)
│   └── processed/        #    파생 패널·모델·시나리오 (01~06이 생성)
├── project_proposal_v3.md
├── requirements.txt
└── README.md
```

---

## 한계

정직하게 밝혀둡니다. 자세한 내용은 보고서 §8을 참조하세요.

- **작물이 2개뿐**입니다. 실제 미국 중서부의 표준 관행은 옥수수–대두 **윤작**이므로, "카운티가 작물 하나를
  고른다"는 단작 가정 자체가 단순화입니다. 결과는 *"이 카운티의 주력을 어느 쪽으로 기울일까"* 로 읽어야 합니다.
- **+2℃는 예측이 아니라 처방된 시나리오**입니다. 34년간 관측된 온난화는 +0.42℃에 불과합니다.
- **가격·변동비는 데이터에 없는 외부 가정**입니다(옥수수 $4.50/bu·$400/ac, 대두 $10.50/bu·$205/ac).
  옥수수 변동비가 대두의 약 2배라는 비대칭이 대두 전환을 만드는 핵심 동력이므로, 결과의 방향이 이 값에
  의존합니다. **USDA ERS로 출처 확정 및 민감도 분석이 필요합니다.**
- **H8(재배 벨트 북상)은 판정 불가**입니다. 모델의 온도 채널이 고온 피해뿐이고, 북상을 만드는
  생육기간 연장·서리일수 감소가 피처셋에 없습니다. 모델이 북상을 *부정한* 것이 아닙니다.
- **예측오차와 의사결정 임계가 같은 자릿수**입니다. 옥수수 RMSE 20 bu/ac ≈ 순이익 $90/ac로, 무릎
  λ(42 $/ac)와 자릿수가 같습니다. 개별 카운티 단위 전환 권고는 노이즈에 취약하며, 지역·위도대 등
  **집계 수준의 결론이 훨씬 견고합니다.**
- **상관이지 인과가 아닙니다.** 고정효과로 지역·연도 교란을 상당 부분 통제했으나 실험적 인과는 아닙니다.

---

## 참고문헌

- Schlenker, W., & Roberts, M. J. (2009). Nonlinear temperature effects indicate severe damages to
  U.S. crop yields under climate change. *PNAS*, 106(37), 15594–15598.
- Metawa, N., Hassan, M. K., & Elhoseny, M. (2017). Genetic algorithm based model for optimizing
  bank lending decisions. *Expert Systems with Applications*, 80, 75–82.
- USDA ERS. *Commodity Costs and Returns* (옥수수·대두 operating cost 기준).
