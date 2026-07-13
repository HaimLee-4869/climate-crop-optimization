# -*- coding: utf-8 -*-
"""
단작(單作) 작물 배분 조합 최적화 — 유전 알고리즘(GA) 구현.

`notebooks/06_optimization.ipynb` 의 GA 로직을 재사용 가능한 모듈로 추출한 것이다.
노트북과 **연산 순서·난수 소비 순서가 동일**하므로, 같은 seed 로 돌리면 결과가 비트 단위로
재현된다.

──────────────────────────────────────────────────────────────────────────────
문제 정의
──────────────────────────────────────────────────────────────────────────────
각 의사결정 단위(카운티) c 가 작물 하나 z_c 를 골라 가용 면적 A_c 전체를 배정한다.

    maximize   Σ_c A_c · v[c, z_c]  −  λ · Σ_{c : z_c ≠ x̄_c} A_c
    s.t.       z_c ∈ {0, ..., K-1}

    v[c,k] : 단위면적당 가치 (예: 순이익 $/ac). 작물 간 **더할 수 있는 단위**여야 한다.
    A_c    : 단위 c 의 가용 면적
    x̄_c    : 현재 배분 (전환비용의 기준)
    λ      : 전환비용 세기. 단위가 v 와 같으므로("$/ac") "작물을 바꾸려면 에이커당
             최소 얼마의 이득이 필요한가"로 해석된다.

──────────────────────────────────────────────────────────────────────────────
중요한 구조적 성질 — 이 문제는 분리 가능(separable)하다
──────────────────────────────────────────────────────────────────────────────
전환비용만 있는 위 문제는 단위 간 제약이 없고 A_c > 0 이므로, 각 단위를 독립적으로
argmax 하면 **정확 최적해**가 O(N) 에 나온다 (`CropAllocationProblem.exact_optimum`).

따라서 이 형태에서 GA 는 "필요"하지 않다. GA 가 실제로 필요해지는 것은 단위를 서로 묶는
제약이 붙을 때이며, 그 예가 최소 수요 제약 `Σ_{z_c=k} A_c ≥ D_k` 다
(`make_min_demand_constraint`). 조합 폭발은 결정변수가 많아서가 아니라 **제약이 변수를
묶을 때** 생긴다.

exact_optimum 이 있다는 것은 약점이 아니라 자산이다 — 메타휴리스틱을 추정 상한이 아니라
진짜 정확해와 채점할 수 있다.

──────────────────────────────────────────────────────────────────────────────
주의 — 목적계수 v 의 단위
──────────────────────────────────────────────────────────────────────────────
작물 간 **수확량(bu/ac)을 그대로 더하면 안 된다.** 옥수수 1부셸과 대두 1부셸은 무게도
가격도 다르다. corn-project 에서 bu/ac 를 그대로 쓰면 2,142개 카운티 전부에서 옥수수가
이겨 최적해가 "모든 카운티 옥수수"로 퇴화(degenerate)한다.
가격·비용을 반영한 순이익($/ac) 등 **공통 단위**로 변환해서 넘길 것.

──────────────────────────────────────────────────────────────────────────────
사용 예
──────────────────────────────────────────────────────────────────────────────
    import sys; sys.path.append('../src')
    from optimize_ga import CropAllocationProblem, run_ga, greedy, warming_loss

    prob = CropAllocationProblem(value=V, area=A, baseline=XBAR)  # V:(N,K) A:(N,) XBAR:(N,)

    z_greedy = greedy(prob)                       # 전환비용 무시 기준선
    z_exact  = prob.exact_optimum(lam=42.0)       # 정확 최적해 (O(N))
    res      = run_ga(prob, lam=42.0, seed=42)    # GA
    print(res['best_fit'], prob.n_switched(res['best']))

GA 는 이진(K=2) 문제에만 적용된다 (bit-flip 돌연변이). 적합도·정확해 관련 함수는 K≥2 를
모두 지원한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    'CropAllocationProblem',
    'run_ga',
    'greedy',
    'no_adaptation',
    'warming_loss',
    'tournament_select',
    'uniform_crossover',
    'bitflip_mutate',
    'init_population',
    'make_min_demand_constraint',
    'sweep_lambda',
    'find_knee',
]


# ══════════════════════════════════════════════════════════════════════════════
# 문제 정의 — 목적함수 / 정확해
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class CropAllocationProblem:
    """단작 배분 문제의 데이터와 목적함수를 함께 들고 있는 객체.

    Parameters
    ----------
    value : ndarray, shape (N, K)
        v[c, k] = 단위 c 에서 작물 k 를 골랐을 때의 단위면적당 가치.
        **작물 간 더할 수 있는 공통 단위**여야 한다 (예: 순이익 $/ac). 모듈 docstring 참조.
    area : ndarray, shape (N,)
        A_c = 각 단위의 가용 면적. 모두 양수여야 한다.
    baseline : ndarray, shape (N,)
        x̄_c = 현재 배분(작물 인덱스). 전환비용의 기준점.
    """

    value: np.ndarray
    area: np.ndarray
    baseline: np.ndarray
    _idx: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.value = np.asarray(self.value, dtype=float)
        self.area = np.asarray(self.area, dtype=float)
        self.baseline = np.asarray(self.baseline).astype(np.int8)

        if self.value.ndim != 2:
            raise ValueError('value 는 (N, K) 2차원이어야 한다. 받은 shape=%r'
                             % (self.value.shape,))
        n, k = self.value.shape
        if self.area.shape != (n,):
            raise ValueError('area shape %r != (%d,)' % (self.area.shape, n))
        if self.baseline.shape != (n,):
            raise ValueError('baseline shape %r != (%d,)' % (self.baseline.shape, n))
        if not np.isfinite(self.value).all():
            raise ValueError('value 에 NaN/Inf 가 있다.')
        if not (self.area > 0).all():
            raise ValueError('area 는 모두 양수여야 한다.')
        if self.baseline.min() < 0 or self.baseline.max() >= k:
            raise ValueError('baseline 의 작물 인덱스가 [0, %d) 범위를 벗어난다.' % k)

        self._idx = np.arange(n)

    # ── 크기 ────────────────────────────────────────────────────────────────
    @property
    def n_units(self) -> int:
        """의사결정 단위 수 N (카운티 수)."""
        return self.value.shape[0]

    @property
    def n_crops(self) -> int:
        """작물 수 K."""
        return self.value.shape[1]

    @property
    def total_area(self) -> float:
        """전체 가용 면적 Σ A_c."""
        return float(self.area.sum())

    # ── 목적함수 ─────────────────────────────────────────────────────────────
    def total_value(self, z: np.ndarray) -> float:
        """총생산 Σ_c A_c · v[c, z_c]. 전환비용은 빼지 않는다.

        Parameters
        ----------
        z : ndarray, shape (N,) — 배분(작물 인덱스 벡터)

        Returns
        -------
        float — 총가치 (value 단위 × 면적 단위, 예: $)
        """
        return float(np.sum(self.area * self.value[self._idx, z]))

    def switch_area(self, z: np.ndarray) -> float:
        """현재 배분 x̄ 와 다르게 고른 단위들의 면적 합 (ac)."""
        return float(self.area[z != self.baseline].sum())

    def n_switched(self, z: np.ndarray) -> int:
        """작물을 바꾼 단위(카운티)의 개수."""
        return int((z != self.baseline).sum())

    def fitness(self, z: np.ndarray, lam: float) -> float:
        """적합도 F_λ(z) = 총생산 − λ · 전환면적.  (개체 1개)

        Parameters
        ----------
        z   : ndarray, shape (N,) — 배분
        lam : float — 전환비용 세기 (v 와 같은 단위, 예: $/ac)

        Returns
        -------
        float
        """
        return self.total_value(z) - lam * self.switch_area(z)

    def fitness_pop(self, Z: np.ndarray, lam: float) -> np.ndarray:
        """개체군 (P, N) 을 한 번에 평가한다. GA 내부용 (벡터화).

        Parameters
        ----------
        Z   : ndarray, shape (P, N) — 개체군
        lam : float — 전환비용 세기

        Returns
        -------
        ndarray, shape (P,) — 각 개체의 적합도
        """
        val = self.value[self._idx[None, :], Z]                    # (P,N) 고른 작물의 v
        prod = (val * self.area[None, :]).sum(axis=1)
        pen = lam * ((Z != self.baseline[None, :]) * self.area[None, :]).sum(axis=1)
        return prod - pen

    def adjusted_value(self, lam: float) -> np.ndarray:
        """전환비용을 흡수한 단위면적당 가치  ṽ[c,k] = v[c,k] − λ·1[k ≠ x̄_c].

        F_λ(z) = Σ_c A_c · ṽ[c, z_c] 이므로, 이 행렬만 있으면 문제가 단위별로 분리된다.

        Returns
        -------
        ndarray, shape (N, K)
        """
        return self.value - lam * (np.arange(self.n_crops)[None, :]
                                   != self.baseline[:, None])

    def exact_optimum(self, lam: float) -> np.ndarray:
        """분리가능성을 이용한 **정확 최적해**. O(N).

        단위 간 제약이 없으므로 각 단위에서 ṽ[c,·] 를 argmax 하면 그것이 전역 최적해다.
        (제약이 붙으면 성립하지 않는다 — `make_min_demand_constraint` 참조)

        Parameters
        ----------
        lam : float — 전환비용 세기

        Returns
        -------
        ndarray, shape (N,), dtype int8 — 최적 배분
        """
        return self.adjusted_value(lam).argmax(axis=1).astype(np.int8)

    def crop_area(self, z: np.ndarray, crop: int) -> float:
        """배분 z 에서 특정 작물에 배정된 총 면적."""
        return float(self.area[z == crop].sum())

    def recovery_pct(self, z: np.ndarray, v_no_adapt: float, loss: float) -> float:
        """무조정 대비 손실 회복률 (%). 100% = 손실을 전부 되찾음.

        Parameters
        ----------
        z          : ndarray — 평가할 배분
        v_no_adapt : float — 무조정(현재 배분 유지) 시의 총가치
        loss       : float — 되찾아야 할 손실 (예: 온난화로 잃은 총가치)

        Returns
        -------
        float — 회복률 (%)

        Notes
        -----
        현재 배분 x̄ 는 충격 이전에도 모델 기준 최적이 아닐 수 있다. 그 경우 회복률에는
        "충격에 대한 적응"과 "원래 있던 비효율 제거"가 섞이며 100% 를 넘을 수도 있다.
        해석 시 이 둘을 분리해 말하는 것이 정직하다.
        """
        return 100.0 * (self.total_value(z) - v_no_adapt) / loss


# ══════════════════════════════════════════════════════════════════════════════
# 기준선
# ══════════════════════════════════════════════════════════════════════════════
def no_adaptation(problem: CropAllocationProblem) -> np.ndarray:
    """기준선 1 — 무조정: 현재 배분 x̄ 를 그대로 유지한다."""
    return problem.baseline.copy()


def greedy(problem: CropAllocationProblem) -> np.ndarray:
    """기준선 2 — Greedy: 전환비용을 무시하고 각 단위에서 가치가 최대인 작물을 고른다.

    λ=0 의 정확 최적해와 동일하며, 전환비용이 있는 문제의 **이론적 상한** 역할을 한다.

    Returns
    -------
    ndarray, shape (N,), dtype int8
    """
    return problem.exact_optimum(lam=0.0)


def warming_loss(problem_before: CropAllocationProblem,
                 problem_after: CropAllocationProblem) -> dict:
    """충격(예: 온난화) 전후의 무조정 총가치와 손실을 계산한다.

    두 problem 은 **같은 area·baseline**, 다른 value 를 갖는다고 가정한다
    (예: baseline 기후 vs +2℃ 기후에서의 순이익).

    Parameters
    ----------
    problem_before : 충격 전 문제
    problem_after  : 충격 후 문제

    Returns
    -------
    dict — v_before, v_after, loss, loss_pct
        loss > 0 이면 충격으로 가치가 줄었다는 뜻.
    """
    z = problem_before.baseline
    v_before = problem_before.total_value(z)
    v_after = problem_after.total_value(z)
    loss = v_before - v_after
    return dict(v_before=v_before, v_after=v_after, loss=loss,
                loss_pct=100.0 * loss / v_before)


# ══════════════════════════════════════════════════════════════════════════════
# GA 연산자 — 선택 / 교차 / 돌연변이 / 초기화
# ══════════════════════════════════════════════════════════════════════════════
def tournament_select(P: np.ndarray, fit: np.ndarray, rng: np.random.Generator,
                      tour_k: int = 3) -> np.ndarray:
    """토너먼트 선택. 무작위로 뽑은 tour_k 명 중 적합도 최고를 부모로 삼는다.

    Parameters
    ----------
    P      : ndarray (P, N) — 개체군
    fit    : ndarray (P,)   — 각 개체의 적합도
    rng    : np.random.Generator
    tour_k : int — 토너먼트 크기 (클수록 선택압이 세다)

    Returns
    -------
    ndarray (P, N) — 선택된 부모 (개체군과 같은 크기)
    """
    pop_size = P.shape[0]
    cand = rng.integers(0, pop_size, size=(pop_size, tour_k))
    return P[cand[np.arange(pop_size), fit[cand].argmax(axis=1)]]


def uniform_crossover(parents: np.ndarray, rng: np.random.Generator,
                      p_cx: float = 0.9) -> np.ndarray:
    """균등 교차(uniform crossover). 유전자마다 50% 확률로 두 부모를 맞바꾼다.

    부모를 (0,1), (2,3), … 짝으로 묶고, 각 짝에 대해 p_cx 확률로 교차를 수행한다.

    Parameters
    ----------
    parents : ndarray (P, N)
    rng     : np.random.Generator
    p_cx    : float — 교차 확률

    Returns
    -------
    ndarray (P, N) — 자식 개체군
    """
    pop_size = parents.shape[0]
    p1, p2 = parents[0::2], parents[1::2]
    m = rng.random(p1.shape) < 0.5                    # 유전자별 교환 마스크
    do_cx = (rng.random((len(p1), 1)) < p_cx)         # 짝별 교차 여부
    c1 = np.where(do_cx & m, p2, p1)
    c2 = np.where(do_cx & m, p1, p2)
    return np.vstack([c1, c2])[:pop_size]


def bitflip_mutate(C: np.ndarray, rng: np.random.Generator, p_mut: float) -> np.ndarray:
    """비트 플립 돌연변이. 각 유전자를 p_mut 확률로 뒤집는다 (0↔1). 이진 문제 전용.

    Parameters
    ----------
    C     : ndarray (P, N)
    rng   : np.random.Generator
    p_mut : float — 유전자당 변이 확률. 보통 1/N (개체당 평균 1개 유전자).

    Returns
    -------
    ndarray (P, N), dtype int8

    Notes
    -----
    corn-project 에서 p_mut 를 1/N → 2/N → 5/N 로 올리면 최적성 갭이
    0.00% → 0.03% → 0.6% 로 **나빠졌다.** 분리가능 문제에서 최적 근처의 돌연변이는
    개선이 아니라 파괴로 작용한다 (탐색 vs 활용의 균형).
    """
    mut = rng.random(C.shape) < p_mut
    return np.where(mut, 1 - C, C).astype(np.int8)


def init_population(problem: CropAllocationProblem, pop_size: int,
                    rng: np.random.Generator, jitter: float = 0.05) -> np.ndarray:
    """초기 개체군. 절반은 현재 배분 x̄ 를 조금 흔든 것, 절반은 완전 무작위.

    현실적 출발점(x̄ 근처)과 탐색 다양성(무작위)을 절반씩 섞어 수렴 속도와 탐색을 균형 맞춘다.

    Parameters
    ----------
    problem  : CropAllocationProblem
    pop_size : int
    rng      : np.random.Generator
    jitter   : float — x̄ 기반 개체에서 각 유전자를 뒤집을 확률

    Returns
    -------
    ndarray (pop_size, N), dtype int8
    """
    n = problem.n_units
    P = np.empty((pop_size, n), dtype=np.int8)
    half = pop_size // 2
    P[:half] = problem.baseline[None, :]
    flip = rng.random((half, n)) < jitter
    P[:half] = np.where(flip, 1 - P[:half], P[:half])
    P[half:] = rng.integers(0, 2, size=(pop_size - half, n), dtype=np.int8)
    return P


# ══════════════════════════════════════════════════════════════════════════════
# GA 메인 루프
# ══════════════════════════════════════════════════════════════════════════════
def run_ga(problem: CropAllocationProblem,
           lam: float,
           pop_size: int = 200,
           n_generations: int = 800,
           mutation_rate: float | None = None,
           crossover_rate: float = 0.9,
           tournament_k: int = 3,
           n_elite: int = 2,
           seed: int = 42,
           feasible: Callable[[np.ndarray], np.ndarray] | None = None,
           repair: Callable[..., np.ndarray] | None = None,
           init: np.ndarray | None = None,
           init_jitter: float = 0.05) -> dict:
    """이진 유전 알고리즘으로 단작 배분 문제를 푼다.

    염색체 = 길이 N 의 이진 벡터(각 유전자 = 한 단위의 작물 선택).
    연산자 = 토너먼트 선택 · 균등 교차 · 비트플립 돌연변이 · 엘리트 보존.

    Parameters
    ----------
    problem        : CropAllocationProblem — K=2(이진) 문제여야 한다
    lam            : float — 전환비용 세기 ($/ac)
    pop_size       : int   — 개체군 크기
    n_generations  : int   — 세대 수
    mutation_rate  : float — 유전자당 변이 확률. None 이면 1/N (개체당 평균 1개).
    crossover_rate : float — 교차 확률
    tournament_k   : int   — 토너먼트 크기
    n_elite        : int   — 세대마다 무손실로 넘길 엘리트 개체 수
    seed           : int   — 난수 시드 (재현성)
    feasible       : callable (P,N) -> (P,) bool, optional
        제약 문제에서 실현가능성 판정. 실현 불가능 개체의 적합도를 −inf 로 만든다.
    repair         : callable ((P,N), rng) -> (P,N), optional
        제약 위반 개체를 실현가능하게 고치는 수리 연산자.
    init           : ndarray (N,), optional — 초기 개체군의 0번 개체로 주입할 시드 해
    init_jitter    : float — 초기화 시 x̄ 를 흔드는 정도

    Returns
    -------
    dict
        best      : ndarray (N,) int8 — 최고 개체
        best_fit  : float            — 그 적합도
        history   : ndarray (n_generations,) — 세대별 최고 적합도 (수렴 곡선)
        n_eval    : int              — 총 적합도 평가 횟수
        seconds   : float            — 소요 시간

    Raises
    ------
    ValueError — 이진 문제(K=2)가 아닐 때

    Notes
    -----
    전환비용만 있는 문제라면 `problem.exact_optimum(lam)` 이 O(N) 에 정확해를 준다.
    GA 는 제약이 붙어 분리가능성이 깨졌을 때 의미가 있다 (`make_min_demand_constraint`).
    """
    if problem.n_crops != 2:
        raise ValueError('run_ga 는 이진(K=2) 문제만 지원한다 (bit-flip 돌연변이). '
                         'K=%d 를 받았다.' % problem.n_crops)

    n = problem.n_units
    rng = np.random.default_rng(seed)
    if mutation_rate is None:
        mutation_rate = 1.0 / n

    # ── 초기화 ──────────────────────────────────────────────────────────────
    P = init_population(problem, pop_size, rng, jitter=init_jitter)
    if init is not None:
        P[0] = init                                   # 시드 해 주입 (선택)
    if repair is not None:
        P = repair(P, rng)

    fit = problem.fitness_pop(P, lam)
    if feasible is not None:
        fit = np.where(feasible(P), fit, -np.inf)

    history = np.empty(n_generations)
    n_eval = pop_size
    t0 = time.perf_counter()

    # ── 세대 루프 ───────────────────────────────────────────────────────────
    for _ in range(n_generations):
        # 엘리트 확보
        elite_idx = np.argsort(fit)[-n_elite:]
        elite, elite_fit = P[elite_idx].copy(), fit[elite_idx].copy()

        # 선택 → 교차 → 돌연변이
        parents = tournament_select(P, fit, rng, tour_k=tournament_k)
        C = uniform_crossover(parents, rng, p_cx=crossover_rate)
        C = bitflip_mutate(C, rng, mutation_rate)

        if repair is not None:
            C = repair(C, rng)

        cf = problem.fitness_pop(C, lam)
        if feasible is not None:
            cf = np.where(feasible(C), cf, -np.inf)
        n_eval += pop_size

        # 엘리트 보존 — 최악 개체를 엘리트로 교체
        worst = np.argsort(cf)[:n_elite]
        C[worst], cf[worst] = elite, elite_fit

        P, fit = C, cf
        history[_] = fit.max()

    seconds = time.perf_counter() - t0
    b = int(fit.argmax())
    return dict(best=P[b].copy(), best_fit=float(fit[b]), history=history,
                n_eval=n_eval, seconds=seconds)


# ══════════════════════════════════════════════════════════════════════════════
# 제약 — 최소 수요 (GA 가 실제로 필요해지는 지점)
# ══════════════════════════════════════════════════════════════════════════════
def make_min_demand_constraint(problem: CropAllocationProblem,
                               lam: float,
                               min_area: float,
                               crop: int = 0):
    """최소 수요 제약  Σ_{c : z_c = crop} A_c ≥ min_area  에 대한 도구 3종을 만든다.

    이 제약은 단위(카운티)를 서로 묶어 **분리가능성을 깨뜨린다.** 문제는
    "목적함수 손실을 최소화하면서 면적 하한을 채우는" 커버링 배낭(covering knapsack)이 되고,
    단위별 argmax 로는 더 이상 풀리지 않는다. 여기서부터 GA 가 제 역할을 한다.

    Parameters
    ----------
    problem  : CropAllocationProblem (K=2)
    lam      : float — 전환비용 세기
    min_area : float — 해당 작물에 배정해야 하는 최소 면적 (A_c 와 같은 단위)
    crop     : int   — 하한을 걸 작물 인덱스 (기본 0 = 옥수수)

    Returns
    -------
    (feasible, repair, lp_upper_bound) : tuple of callables

        feasible(Z) -> (P,) bool
            개체군의 각 개체가 제약을 만족하는지.

        repair(Z, rng=None) -> (P, N) int8
            제약 위반 개체를 고친다. 목적함수 손실이 **싼 단위부터** 대상 작물로 뒤집어
            하한을 채운다 (탐욕적 수리). 벡터화되어 있다.

        lp_upper_bound() -> (float, bool)
            (상한값, 제약이 비구속인지 여부).
            커버링 배낭의 **LP 완화**(마지막 항목만 분수 허용)를 풀어 정수 최적해의
            상한을 준다. GA 의 **증명 가능한 최적성 갭**을 계산하는 데 쓴다:
                gap = (upper_bound − ga_fitness) / |upper_bound|

    Examples
    --------
    >>> feas, rep, lub = make_min_demand_constraint(prob, lam=42.0,
    ...                                             min_area=0.4 * prob.total_area)
    >>> ub, is_slack = lub()
    >>> res = run_ga(prob, lam=42.0, feasible=feas, repair=rep,
    ...              init=rep(prob.exact_optimum(42.0)[None, :], None)[0])
    >>> gap = 100 * (ub - res['best_fit']) / abs(ub)
    """
    if problem.n_crops != 2:
        raise ValueError('현재 구현은 이진(K=2) 문제만 지원한다.')

    A = problem.area
    n = problem.n_units
    other = 1 - crop                       # 뒤집어 올 수 있는 반대편 작물

    def target_area(Z: np.ndarray) -> np.ndarray:
        """개체군 각 개체에서 대상 작물에 배정된 면적 (P,)."""
        return ((Z == crop) * A[None, :]).sum(axis=1)

    def feasible(Z: np.ndarray) -> np.ndarray:
        return target_area(Z) >= min_area - 1e-6

    # other → crop 전환의 단위면적당 목적함수 손실. 작을수록 싸게 하한을 채운다.
    adj = problem.adjusted_value(lam)                         # (N, 2)
    loss_per_ac = adj[:, other] - adj[:, crop]
    order = np.argsort(loss_per_ac)                           # 싼 것부터
    A_ordered = A[order]

    def repair(Z: np.ndarray, rng=None) -> np.ndarray:
        """하한 미달 개체를, 손실이 싼 단위부터 대상 작물로 뒤집어 고친다. (벡터화)"""
        need = min_area - target_area(Z)                      # (P,) 부족분
        if not (need > 0).any():
            return Z
        Zo = Z[:, order]                                      # 손실이 싼 순서로 재배열
        is_other = (Zo == other)
        csum = np.cumsum(is_other * A_ordered[None, :], axis=1)   # 뒤집으면 얻는 누적 면적
        prev = csum - is_other * A_ordered[None, :]               # 직전까지의 누적
        flip = is_other & (prev < np.maximum(need, 0)[:, None])   # 부족분을 채울 때까지만
        Zo = np.where(flip, crop, Zo).astype(np.int8)
        Z = Z.copy()
        Z[:, order] = Zo
        return Z

    def lp_upper_bound():
        """분수 허용 커버링 배낭 → 정수 최적해의 상한."""
        z0 = adj.argmax(axis=1)                               # 무제약 최적
        shortfall = min_area - A[z0 == crop].sum()
        base = float((A * adj[np.arange(n), z0]).sum())
        if shortfall <= 0:
            return base, True                                 # 제약 비구속
        loss, filled = 0.0, 0.0
        for c in order:                                       # 손실이 싼 단위부터
            if z0[c] != other:
                continue
            if filled >= shortfall:
                break
            take = min(A[c], shortfall - filled)              # 분수 허용 (완화)
            loss += take * max(loss_per_ac[c], 0.0)
            filled += take
        return base - loss, False

    return feasible, repair, lp_upper_bound


# ══════════════════════════════════════════════════════════════════════════════
# 전환비용 트레이드오프 분석
# ══════════════════════════════════════════════════════════════════════════════
def sweep_lambda(problem: CropAllocationProblem,
                 lambdas: Sequence[float],
                 v_no_adapt: float,
                 loss: float,
                 use_ga: bool = False,
                 **ga_kwargs) -> list:
    """전환비용 λ 를 스윕하며 각 λ 의 최적해와 그 특성을 기록한다.

    x축(바꾼 면적 %) – y축(손실 회복률 %) 트레이드오프 곡선의 재료가 된다.

    Parameters
    ----------
    problem    : CropAllocationProblem
    lambdas    : λ 값들의 순열
    v_no_adapt : float — 무조정 시 총가치 (회복률의 기준점)
    loss       : float — 되찾아야 할 손실
    use_ga     : bool  — True 면 GA 로, False 면 정확해로 푼다 (기본: 정확해, 훨씬 빠름)
    **ga_kwargs : use_ga=True 일 때 run_ga 로 넘길 인자

    Returns
    -------
    list[dict] — λ 별로 lam, total_value, fitness, n_switch, pct_units,
                 pct_area, recovery_pct 를 담은 레코드
    """
    rows = []
    for lam in lambdas:
        z = run_ga(problem, lam, **ga_kwargs)['best'] if use_ga else problem.exact_optimum(lam)
        rows.append(dict(
            lam=float(lam),
            total_value=problem.total_value(z),
            fitness=problem.fitness(z, lam),
            n_switch=problem.n_switched(z),
            pct_units=100.0 * problem.n_switched(z) / problem.n_units,
            pct_area=100.0 * problem.switch_area(z) / problem.total_area,
            recovery_pct=problem.recovery_pct(z, v_no_adapt, loss),
        ))
    return rows


def find_knee(x: np.ndarray, y: np.ndarray) -> int:
    """트레이드오프 곡선의 무릎(knee) 인덱스를 찾는다.

    표준 knee/elbow 규칙: 곡선의 **양 끝점을 잇는 직선에서 수직 거리가 최대인 점.**
    "적은 변화로 큰 개선"을 얻는 스윗스팟에 해당한다.

    Parameters
    ----------
    x, y : ndarray — 곡선의 좌표. x 기준으로 정렬되어 있다고 가정한다.

    Returns
    -------
    int — 무릎에 해당하는 인덱스
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    d = p1 - p0
    d = d / np.linalg.norm(d)
    pts = np.column_stack([x, y]) - p0
    dist = np.abs(pts[:, 0] * d[1] - pts[:, 1] * d[0])   # 직선까지의 수직 거리
    return int(dist.argmax())
