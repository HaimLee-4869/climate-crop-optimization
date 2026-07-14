# 기후변화 하 미국 곡물 수확량 예측과 3작물 배분 최적화

미국 카운티 단위 기후·토양 자료로 **옥수수·대두·밀** 수확량을 예측하고, 그 예측치를 계수로 삼아
**+2℃ 온난화 시나리오 하의 작물 배분을 조합 최적화(GA · SA · TS)로** 푸는 프로젝트입니다.

> **핵심 메시지 — "더위가 옥수수를 해치고, 적은 재배치로 손실을 상당 부분 만회할 수 있다.
> 그리고 그 재배치의 방향(남부 → 밀)은 가격 가정이 아니라 작물의 고온 반응에서 나온다."**

수확량 예측 모델 3개(옥수수·대두·밀) + 기후 시나리오 + 메타휴리스틱 최적화 3종으로 구성됩니다.
데이터는 미국 카운티 단위 농업·기후 패널(1981–2015), 2,644개 카운티 · 70,721개 관측입니다.

---

## 주요 결과

### 예측 모델 (LightGBM, GroupKFold(stco) 5-fold)

| 작물 | CV R² | RMSE | 학습 연도 |
|---|---|---|---|
| 옥수수 | **0.732** | 20.08 bu/ac | 1981–2015 |
| 대두 | **0.807** | 4.62 bu/ac | 1981–2015 |
| 밀 | **0.691** | 8.78 bu/ac | 1981–2007 |

### 고온 피해 — 작물 간 내열성 서열 (카운티·연도 고정효과)

29℃ 이하 노출 1일을 29℃ 초과로 대체할 때의 효과:

| 작물 | 대체효과 | t | 평균 대비 |
|---|---|---|---|
| 옥수수 | **−1.67** bu/ac/일 | −60.7 | −1.54%/일 |
| 대두 | −0.39 bu/ac/일 | −51.7 | −1.17%/일 |
| 밀 | −0.17 bu/ac/일 | −20.9 | **−0.40%/일** |

밀은 29℃ 초과 노출의 **직접 효과가 양수**(b_above = +0.069)입니다. 이 한 줄이 아래 최적화에서
"남부가 밀로 전환된다"는 결론의 물리적 근거입니다.

### 최적화 (+2℃, 전환비용 무릎 λ = 42.29 $/ac)

| 항목 | 결과 |
|---|---|
| +2℃ 무조정 영향 | 옥수수 수확량 **−8.98%**, 카운티의 79.2%가 손해 (South −12.1%로 최악) |
| 2작물 재배치 (06) | 농지 11.2%만 바꿔 손실의 **15.99% 회복** (238 / 2,142 카운티) |
| Greedy 대비 | Greedy는 **2.7배 많은 면적**(29.9%)을 갈아엎고 22.6% 회복 |
| **3작물 재배치 (10)** | 회복률 **19.12%** (+3.13%p) — 밀 도입만으로 개선 |
| **재배 벨트 재편** | 밀이 **남부**에서 선택 (TX·OK·VA·MO·AR·GA) — 213 카운티, 평균 위도 36.1° (전국 38.3°) |
| **식량안보 제약 (11)** | 옥수수 생산 100% 유지 시 회복률 19.12% → **17.35%** (−1.76%p). 그림자 가격 μ* = 0.173 $/bu |
| **USDA 실측가 (12)** | 하드코딩 → USDA ERS 2025 교체 시 회복률 16.65%, 밀 카운티 213 → **419** |
| **가격 ±30% 민감도** | 15개 시나리오 **전부**에서 "남부 = 밀" 유지 → 결론은 가격이 아닌 기후 반응에서 나옴 |

### 방법 비교 (GA · SA · TS)

| 문제 | GA | SA | TS |
|---|---|---|---|
| 2작물 (06–09) — 분리 가능 | 갭 0% | 갭 0% | 갭 0% |
| 3작물 (10) — 분리 가능 | 0.0104% | 0.000003% | **0%** |
| 3작물 + 최소수요 (11) — **NP-hard** | **0.0120%** (불안정) | 0.0140% (안정) | 0.0140% (안정, 최속) |

전환비용만 있는 문제는 카운티별로 **분리 가능**해 정확해가 O(N·K)에 나옵니다. 메타휴리스틱이
실제로 필요해지는 지점은 **최소 수요 제약**이 카운티를 서로 묶어 분리가능성을 깨뜨렸을 때이며,
그때 비로소 세 방법 모두 0이 아닌 갭을 갖습니다. (11에서는 GA가 평균 갭은 낮지만 seed마다 흔들리고,
SA/TS는 10개 seed 전부 동일한 해로 수렴합니다.)

**독립 검증(09)**: 완전탐색 2¹⁸ 전수 열거, 목적함수 손계산, Greedy 상한 도달, 무작위 2,000개 대조
(z = 29σ) 등 **11개 항목 전부 통과**.

**핵심 통찰 — 평균기온이 아니라 온도 노출 분포가 문제다.** 노출가중 평균기온이 사실상 동일한
(20.51℃ vs 20.52℃) 두 카운티·연도가, 29℃ 초과 노출일수는 44.8일 대 7.3일로 갈리고
수확량은 95.4 대 153.0 bu/ac로 갈립니다.

전체 분석은 **[`outputs/corn_project_report_final.docx`](outputs/corn_project_report_final.docx)**
(최종 보고서 — 그림 26 · 표 19)와 **[`project_proposal_v3.md`](project_proposal_v3.md)**
(기획·정식화)를 참조하세요.

---

## 파이프라인

노트북은 **번호 순서대로 실행**해야 합니다. 각 단계가 앞 단계의 산출물(`data/processed/`)에 의존합니다.

| # | 노트북 | 하는 일 | 주요 산출물 |
|---|---|---|---|
| 01 | `01_preprocessing.ipynb` | 원자료를 `stco`(+`year`)로 병합, 결측 처리 | `corn_panel.parquet` (70,721 × 136) |
| 02 | `02_eda.ipynb` | 29℃ 임계 신호, 다중공선성, 매칭 사례 | fig01–fig09 |
| 03 | `03_features.ipynb` | 피처 엔지니어링 (전략 A: 도메인 압축 / B: 원자료+정규화) | `features_stratA/B.parquet` |
| 04 | `04_yield_model.ipynb` | **옥수수** 9개 모델 × 2전략 + 고정효과 해석 모델 | `yield_model_final.joblib`, fig10–fig16 |
| 04b | `04b_soybean_model.ipynb` | **대두** 모델 (동일 파이프라인) | `soybean_model_final.joblib`, fig23–fig26 |
| 04c | `04c_wheat_model.ipynb` | **밀** 모델 (동일 파이프라인, 1981–2007) | `wheat_model_final.joblib`, fig41–fig42 |
| 05 | `05_climate_model.ipynb` | 기후 추세 + **+2℃ 시나리오 생성** (184일 제약 보존) | `scenarios.parquet`, fig17–fig22 |
| 06 | `06_optimization.ipynb` | 2작물 단작 조합 최적화 **(GA)** + 전환비용 트레이드오프 | `optimization_results.parquet`, fig27–fig31 |
| 07 | `07_sa_optimization.ipynb` | **Simulated Annealing** + GA 비교 | fig32–fig34 |
| 08 | `08_ts_optimization.ipynb` | **Tabu Search** + GA·SA 3방향 비교 | fig35–fig38 |
| 09 | `09_optimization_validation.ipynb` | **독립 검증 11항목** (완전탐색·손계산·상한·무작위 대조) | `validation_*.csv`, fig39–fig40 |
| 10 | `10_optimization_3crop.ipynb` | **3작물 최적화** (옥수수·대두·밀, 2,056 카운티) | `optimization_3crop_results.parquet`, fig43–fig45 |
| 11 | `11_optimization_mindemand.ipynb` | **최소 수요 제약** (식량안보 ↔ 적응 trade-off, NP-hard) | `mindemand_*.csv`, fig46–fig48 |
| 12 | `12_usda_prices.ipynb` | **USDA ERS 실측 가격** 교체 + 가격 ±30% 민감도 | `usda_*.csv`, fig49–fig50 |

> **의존관계:** `04b`·`04c`는 `04`의 피처 정의에, `05`는 `04`의 모델에, `06`은 `04`·`04b`·`05`에,
> `07`–`09`는 `06`에, `10`은 `04`·`04b`·`04c`·`05`에, `11`·`12`는 `10`의 parquet에 의존합니다.
> `12`는 추가로 `data/raw/`의 USDA CSV 3개가 필요합니다.

### 방법론 요지

- **평가는 `GroupKFold(stco)` 5-fold.** 무작위 분할은 같은 카운티의 다른 연도가 학습·평가에 동시에
  들어가 공간 정보가 샙니다(RF 기준 R² 0.654 → 0.448). 전 과정에서 카운티 단위 분할을 씁니다.
  모델의 목적은 **미래 예측이 아니라 기후–수확량 관계의 학습**이며, 학습된 관계를 시나리오 평가에만 씁니다.
- **184일 제약.** 온도 노출 121구간의 행별 합은 항상 184일(생육기간)입니다. 따라서 개별 계수가 아니라
  **대체효과**(29℃ 이하 1일 → 29℃ 초과 1일)로만 해석해야 합니다.
- **목적함수는 변동비 차감 마진 $/ac.** 옥수수 부셸과 대두 부셸은 더할 수 없는 단위입니다. bu/ac를
  그대로 쓰면 모든 카운티에서 옥수수가 이겨 최적화가 퇴화합니다. 따라서
  `v[c,k] = P_k · ŷ[c,k] − C_k` 로 화폐 단위화합니다.
  `C_k`는 **변동비(operating cost)만** 포함하고 토지임차료·감가상각 같은 고정비는 제외하므로,
  이 값은 **"순이익"이 아니라 변동비 차감 마진(gross margin)** 입니다.
- **분리가능성.** 전환비용만 있는 문제는 카운티 간 제약이 없어 카운티별 argmax가 곧 전역 최적해입니다
  (09에서 2¹⁸ 완전탐색으로 검증). 조합 폭발은 결정변수가 많아서가 아니라 **제약이 변수를 묶을 때** 생깁니다.

---

## 실행법

### 1. 환경

Python 3.11 기준입니다.

```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. 데이터 내려받기 (별도 필요)

**원본 데이터는 이 저장소에 포함되어 있지 않습니다** (약 350MB, 재배포 라이선스 문제).
아래 두 묶음을 `data/raw/` 에 직접 넣어야 합니다.

#### (a) 기후·토양·수확량 — ACDC 계열

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

Schlenker & Roberts (2009, PNAS)의 **Agricultural Climate Data Collection (ACDC)** 계열
데이터셋입니다. 자료 정의는 `data/raw/Manual_ACDCv1_20170814.pdf` 를 참조하세요.
공통 키는 `stco`(5자리 FIPS = 주 2자리 + 카운티 3자리)이고, 시계열 파일은 `year`(1981–2015)를
추가 키로 갖습니다.

#### (b) 가격·변동비 — USDA ERS Commodity Costs and Returns

노트북 `12`가 사용합니다 (`06`–`11`은 하드코딩 값으로도 실행됩니다).

- 출처: **USDA ERS — Commodity Costs and Returns**
- URL: <https://www.ers.usda.gov/data-products/commodity-costs-and-returns>
- **"Recent Cost and Returns"** 섹션에서 **Corn / Soybeans / Wheat** 의 CSV를 각각 내려받아
  아래 이름으로 배치합니다.

```
data/raw/
├── CornCostReturn.csv       # 옥수수
├── SoybeanCostReturn.csv    # 대두
└── WheatCostReturn.csv      # 밀
```

`12`는 각 파일에서 **최신 연도(2025) · `Region = "U.S. total"`** 의 두 항목만 추출합니다.

| 항목 | `Item` | 단위 |
|---|---|---|
| 가격 | `Price` | dollars per bushel at harvest |
| 변동비 | `Total, operating costs` | dollars per planted acre |

추출된 값 (2025, U.S. total):

| 작물 | 가격 ($/bu) | 변동비 ($/ac) |
|---|---|---|
| 옥수수 | 3.93 | 440.11 |
| 대두 | 9.76 | 244.80 |
| 밀 | 5.05 | 155.52 |

> `data/raw/` 는 **읽기 전용**입니다. 모든 파생 산출물은 `data/processed/` 에 씁니다.

### 3. 노트북 실행

```bash
jupyter lab   # 01 → 02 → 03 → 04 → 04b → 04c → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12
```

전체를 한 번에 돌리려면:

```bash
cd notebooks
for nb in 01_preprocessing 02_eda 03_features \
          04_yield_model 04b_soybean_model 04c_wheat_model \
          05_climate_model 06_optimization \
          07_sa_optimization 08_ts_optimization 09_optimization_validation \
          10_optimization_3crop 11_optimization_mindemand 12_usda_prices; do
  py -3.11 -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 "$nb.ipynb"
done
```

> `04`·`04b`·`04c`(각 트리 모델 18개 조합)와 `06`(λ 스윕)이 가장 오래 걸립니다. 각각 30분 내외를
> 예상하세요. 모든 메타휴리스틱은 `seed=42`로 고정되어 재현 가능합니다.

---

## 저장소 구조

```
corn-project/
├── notebooks/            # 01~12 분석 노트북 (실행 결과·그림 포함, 14개)
├── src/
│   ├── optimize_ga.py         # 문제 정의 + 정확해 + GA (2작물, 이진)
│   ├── optimize_sa.py         # Simulated Annealing (2작물)
│   ├── optimize_ts.py         # Tabu Search (2작물)
│   ├── optimize_kcrop.py      # K작물(K≥2) 일반화 — GA·SA·TS
│   └── optimize_mindemand.py  # 최소 수요 제약 + 라그랑주 상한 + GA·SA·TS
├── outputs/
│   ├── figures/          # fig01 ~ fig50 (50개)
│   ├── results/          # 모델 비교·검증·λ 스윕·민감도 등 표 (CSV 41개)
│   ├── corn_project_report.docx        # 초기 보고서 (2작물)
│   └── corn_project_report_final.docx  # ★ 최종 보고서 (3작물 전체)
├── data/                 # ⚠ 저장소에 미포함 — 직접 준비
│   ├── raw/              #    원본 (읽기 전용)
│   └── processed/        #    파생 패널·모델·시나리오 (01~12가 생성)
├── project_proposal_v3.md
├── requirements.txt
└── README.md
```

---

## 한계

정직하게 밝혀둡니다. 자세한 내용은 최종 보고서 §9를 참조하세요.

- **+2℃는 예측이 아니라 처방된 시나리오**입니다. 34년간 관측된 온난화는 +0.42℃에 불과해
  데이터로 미래를 예측할 수 없음을 확인했고, 그래서 **IPCC 수준의 기준선 시나리오** 방식을 채택했습니다.
  "2℃ 오르면 어떻게 되는가"라는 조건부 질문에 답할 뿐, "2℃ 오를 것이다"라고 주장하지 않습니다.
- **가격은 USDA ERS 2025 단일 연도 값**입니다. 2025년은 곡물가 하락과 투입재가 상승이 겹쳐
  **옥수수에 유난히 불리한 해**입니다(옥수수 마진 ≈ $28/ac vs 대두 $102 · 밀 $67).
  이 편향이 결론을 좌우하지 않음은 **가격 ±30% 민감도(12)로 검증**했으나, 여러 해 평균이 더 안전한
  대안입니다.
- **작물별 면적 비율은 점추정치로 신뢰할 수 없습니다.** 가격 ±30%에서 옥수수 0.0~81.7%,
  대두 1.6~95.8%, 밀 2.9~49.2%까지 움직입니다(폭 46~94%p). 반드시 **범위(밴드)로** 제시해야 합니다.
  반면 **배치 패턴(남부 = 밀)은 15개 시나리오 전부에서 유지**되므로 단정할 수 있습니다.
- **현재 배분 x̄는 실제 재배면적이 아닙니다.** 작물별 재배면적 자료가 없어 "관측 수확량 × 가격의
  argmax"로 근사했습니다. 따라서 x̄는 현재 재배 현황이 아니라 **"현재 가격에서 각 카운티가 골랐어야 할
  작물"** 로 읽어야 하며, 회복률에는 "온난화 적응"과 "원래 있던 배분 비효율의 제거"가 섞여 있습니다.
- **밀 데이터가 2007년까지입니다.** 밀 모델은 1981–2007만 학습했고, x̄ 산정에서도 최근 5년 관측이
  없어 전체 기간 평균으로 대체됩니다. 온도 노출창도 3–8월(옥수수 기준)이라 겨울밀의 생육창과
  어긋납니다. 즉 **밀은 과소평가된 상태로 경쟁**하므로, "밀이 선택된다"는 결론은 보수적입니다.
- **단작 가정 자체가 단순화**입니다. 미국 중서부의 표준 관행은 옥수수–대두 **윤작**이므로,
  결과는 *"이 카운티의 주력을 어느 쪽으로 기울일까"* 로 읽어야 합니다.
- **마진은 순이익이 아닙니다.** ERS의 `Total, operating costs`만 차감했고 고정비(토지임차료·감가상각)는
  제외했습니다. 정부보조금·작물보험·윤작의 농학적 이득도 모델에 없습니다.
- **H8(재배 벨트 북상)은 판정 불가**입니다. 모델의 온도 채널이 고온 피해뿐이고, 북상을 만드는
  생육기간 연장·서리일수 감소가 피처셋에 없습니다. 모델이 북상을 *부정한* 것이 아닙니다.
  다만 3작물 확장은 부분적인 답을 줍니다 — 북쪽 경계는 판정 못 해도 **남쪽 경계의 밀 재편은 확인**됐습니다.
- **예측오차와 의사결정 임계가 같은 자릿수**입니다. 옥수수 RMSE 20 bu/ac ≈ 마진 $90/ac로, 무릎
  λ(42.29 $/ac)와 자릿수가 같습니다. 개별 카운티 단위 전환 권고는 노이즈에 취약하며, 지역·위도대 등
  **집계 수준의 결론이 훨씬 견고합니다.**
- **강수는 시나리오 간 고정**했습니다. 온난화에 수반되는 강수 패턴 변화를 모델링하지 않았습니다.
- **상관이지 인과가 아닙니다.** 고정효과로 지역·연도 교란을 상당 부분 통제했으나 실험적 인과는 아닙니다.

---

## 라이선스

코드·노트북·문서는 **[MIT License](LICENSE)** 를 따릅니다.

단, **데이터는 이 라이선스에 포함되지 않습니다.** ACDC 기후·토양·수확량 자료와 USDA ERS
Commodity Costs and Returns는 이 저장소에 재배포되지 않으며, 각자의 이용 약관을 따릅니다.
직접 내려받아 사용하세요 (위 [데이터 내려받기](#2-데이터-내려받기-별도-필요) 참조).

---

## 참고문헌

- Schlenker, W., & Roberts, M. J. (2009). Nonlinear temperature effects indicate severe damages to
  U.S. crop yields under climate change. *PNAS*, 106(37), 15594–15598.
- Metawa, N., Hassan, M. K., & Elhoseny, M. (2017). Genetic algorithm based model for optimizing
  bank lending decisions. *Expert Systems with Applications*, 80, 75–82.
- USDA Economic Research Service. *Commodity Costs and Returns.*
  <https://www.ers.usda.gov/data-products/commodity-costs-and-returns>
  (2025, U.S. total — 옥수수·대두·밀의 `Price` 및 `Total, operating costs`)
