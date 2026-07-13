# -*- coding: utf-8 -*-
"""
단작 작물 배분 조합 최적화 — 담금질 기법(Simulated Annealing, SA) 구현.

개요 v3 §5 Phase 5 의 **A→C 확장 2단계**. `optimize_ga.py` 와 **완전히 같은 문제**를 풀어
GA 와 공정하게 비교하기 위한 모듈이다. 문제 정의·목적함수·전환비용은
`optimize_ga.CropAllocationProblem` 을 그대로 재사용한다 (재정의하지 않는다).

──────────────────────────────────────────────────────────────────────────────
SA 설계
──────────────────────────────────────────────────────────────────────────────
상태(state)   : 길이 N 의 이진 벡터 z. GA 의 염색체와 **같은 표현**.
에너지(energy): E(z) = −F_λ(z).  SA 는 최소화하므로 적합도의 부호를 뒤집는다.
이웃(neighbor): 무작위로 고른 `n_flips` 개 카운티의 작물을 뒤집는다.
수용(accept)  : ΔE ≤ 0 이면 항상 수용. ΔE > 0 이면 확률 exp(−ΔE / T) 로 수용
                (Metropolis 기준) — 이것이 지역 최적을 탈출하는 장치다.
냉각(cooling) : 지수(기하) 냉각  T ← α·T.  온도 레벨마다 일정 횟수 이동을 시도한다.
초기온도 T0   : 무작위 이동을 시험 삼아 던져 **악화 이동의 평균 |ΔE|** 를 재고,
                목표 초기 수용확률 p0 에서 역산한다:  T0 = mean(ΔE_worse) / (−ln p0).
                (강의 자료의 표준 방식. 문제 스케일이 바뀌어도 자동으로 맞춰진다.)

──────────────────────────────────────────────────────────────────────────────
증분 평가 (incremental evaluation) — SA 가 GA 보다 빠른 진짜 이유
──────────────────────────────────────────────────────────────────────────────
목적함수가 카운티별로 분리되므로

    F_λ(z) = Σ_c A_c · ṽ[c, z_c],      ṽ[c,k] = v[c,k] − λ·1[k ≠ x̄_c]

카운티 c 하나를 z_c → z_c' 로 뒤집을 때의 변화량은

    ΔF = A_c · ( ṽ[c, z_c'] − ṽ[c, z_c] )

로 **O(1)** 에 나온다. 전체를 다시 더할 필요가 없다. GA 는 개체 하나를 평가할 때마다
O(N) 이 드는 반면 SA 의 한 이동은 O(1) 이다 — 벽시계 시간 비교를 읽을 때 이 비대칭을
반드시 감안해야 한다 (§ 비교 시 주의).

──────────────────────────────────────────────────────────────────────────────
비교 시 주의
──────────────────────────────────────────────────────────────────────────────
"평가 횟수"를 축으로 GA 와 SA 를 나란히 놓는 것은 메타휴리스틱 문헌의 관례지만,
여기서는 **한 번의 평가 비용이 서로 다르다**(GA: O(N), SA: O(1)). 따라서
  · 해의 품질  → 정확해(exact_optimum) 대비 갭으로 비교하면 공정하다.
  · 계산 시간  → 벽시계로 비교하되, SA 의 이점 상당 부분이 증분 평가에서 온다고 밝힌다.
  · 안정성      → 여러 seed 로 반복해 분산을 본다.

──────────────────────────────────────────────────────────────────────────────
사용 예
──────────────────────────────────────────────────────────────────────────────
    from optimize_ga import CropAllocationProblem
    from optimize_sa import run_sa

    prob = CropAllocationProblem(value=V, area=A, baseline=XBAR)
    res  = run_sa(prob, lam=42.0, n_iter=160_000, alpha=0.97, seed=42)
    print(res['best_fit'], res['n_accept'], res['seconds'])
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from optimize_ga import CropAllocationProblem

__all__ = [
    'energy',
    'estimate_T0',
    'run_sa',
    'sweep_lambda_sa',
]


# ══════════════════════════════════════════════════════════════════════════════
# 에너지 (= −적합도)
# ══════════════════════════════════════════════════════════════════════════════
def energy(problem: CropAllocationProblem, z: np.ndarray, lam: float) -> float:
    """SA 의 에너지 E(z) = −F_λ(z).

    SA 는 에너지를 **최소화**하므로 적합도의 부호를 뒤집는다. 그 외에는 GA 의 적합도와
    완전히 동일한 함수다 (같은 문제를 푼다는 보장).

    Parameters
    ----------
    problem : CropAllocationProblem
    z       : ndarray (N,) — 상태(배분)
    lam     : float — 전환비용 세기

    Returns
    -------
    float — 에너지 (작을수록 좋다)
    """
    return -problem.fitness(z, lam)


def _delta_fitness(adj: np.ndarray, area: np.ndarray, z: np.ndarray,
                   idx: np.ndarray) -> float:
    """idx 위치의 작물을 뒤집었을 때의 **적합도 변화량** ΔF. O(len(idx)).

    목적함수의 분리가능성을 이용한 증분 평가. 전체 재계산과 수학적으로 동일하다.

    Parameters
    ----------
    adj  : ndarray (N, 2) — 전환비용을 흡수한 단위면적당 가치 ṽ (problem.adjusted_value(lam))
    area : ndarray (N,)   — A_c
    z    : ndarray (N,)   — 현재 상태
    idx  : ndarray (k,)   — 뒤집을 카운티 인덱스 (중복 없어야 정확)

    Returns
    -------
    float — ΔF (양수면 적합도가 좋아진다)
    """
    cur = z[idx]
    new = 1 - cur
    return float(np.sum(area[idx] * (adj[idx, new] - adj[idx, cur])))


# ══════════════════════════════════════════════════════════════════════════════
# 초기 온도
# ══════════════════════════════════════════════════════════════════════════════
def estimate_T0(problem: CropAllocationProblem, lam: float, z0: np.ndarray,
                rng: np.random.Generator, target_accept: float = 0.8,
                n_probe: int = 2000, n_flips: int = 1) -> float:
    """목표 초기 수용확률 p0 로부터 초기 온도 T0 를 역산한다.

    무작위 이웃 이동을 `n_probe` 번 시험 삼아 던져 **악화 이동**(ΔE > 0)의 평균 크기를 잰 뒤

        T0 = mean(ΔE_worse) / (−ln p0)

    로 정한다. Metropolis 수용확률 exp(−ΔE/T0) 의 평균이 대략 p0 가 되도록 맞추는 것이다.
    문제의 스케일(여기서는 $ 단위 목적함수)이 바뀌어도 자동으로 적응한다.

    Parameters
    ----------
    problem       : CropAllocationProblem
    lam           : float
    z0            : ndarray (N,) — 탐침을 던질 기준 상태
    rng           : np.random.Generator
    target_accept : float — 목표 초기 수용확률 p0 (0 < p0 < 1). 보통 0.8.
    n_probe       : int   — 시험 이동 횟수
    n_flips       : int   — 이웃 정의(한 번에 뒤집을 카운티 수)

    Returns
    -------
    float — T0 (> 0)

    Notes
    -----
    악화 이동이 하나도 안 나오면(이미 최적이거나 매우 평탄) 1.0 을 돌려준다.
    """
    if not (0.0 < target_accept < 1.0):
        raise ValueError('target_accept 는 (0, 1) 이어야 한다.')

    n = problem.n_units
    adj = problem.adjusted_value(lam)
    area = problem.area

    worse = []
    for _ in range(n_probe):
        idx = rng.choice(n, size=n_flips, replace=False)
        dE = -_delta_fitness(adj, area, z0, idx)     # 에너지 변화 = −ΔF
        if dE > 0:
            worse.append(dE)

    if not worse:
        return 1.0
    return float(np.mean(worse) / (-np.log(target_accept)))


# ══════════════════════════════════════════════════════════════════════════════
# SA 메인 루프
# ══════════════════════════════════════════════════════════════════════════════
def run_sa(problem: CropAllocationProblem,
           lam: float,
           n_iter: int = 160_000,
           n_temp_levels: int = 200,
           alpha: float = 0.97,
           T0: float | None = None,
           target_accept: float = 0.8,
           n_flips: int = 1,
           seed: int = 42,
           init: np.ndarray | None = None,
           feasible_state: Callable[[np.ndarray], bool] | None = None,
           n_records: int = 800) -> dict:
    """담금질 기법(SA)으로 단작 배분 문제를 푼다.

    GA(`optimize_ga.run_ga`)와 **같은 문제·같은 목적함수**를 풀며, 비교를 위해 반환 형식도
    최대한 맞췄다.

    Parameters
    ----------
    problem        : CropAllocationProblem — K=2(이진) 문제
    lam            : float — 전환비용 세기 ($/ac)
    n_iter         : int   — 총 이동 시도 횟수 (= 목적함수 평가 횟수)
    n_temp_levels  : int   — 온도 레벨 수. 레벨마다 n_iter // n_temp_levels 번 이동을 시도한다.
    alpha          : float — 지수 냉각 계수. 레벨이 끝날 때마다 T ← α·T (0 < α < 1).
    T0             : float — 초기 온도. None 이면 `estimate_T0` 로 자동 설정.
    target_accept  : float — T0 자동 설정 시의 목표 초기 수용확률 p0.
    n_flips        : int   — 이웃 정의: 한 번에 뒤집을 카운티 수.
    seed           : int   — 난수 시드 (재현성)
    init           : ndarray (N,) — 초기 상태. None 이면 현재 배분 x̄ 에서 출발.
    feasible_state : callable (N,) -> bool, optional
        제약 문제에서 상태의 실현가능성 판정. 실현 불가능한 이웃은 **거부**한다.
    n_records      : int   — 수렴 곡선에 기록할 점의 개수 (GA 의 세대 수와 맞추면 겹쳐 그리기 편하다)

    Returns
    -------
    dict
        best        : ndarray (N,) int8 — 최고 상태
        best_fit    : float — 그 적합도 F_λ (에너지가 아니라 **적합도**. GA 와 부호를 맞췄다)
        history     : ndarray (n_records,) — 기록 시점별 best-so-far 적합도
        hist_evals  : ndarray (n_records,) — 각 기록 시점의 누적 평가 횟수
        hist_cur    : ndarray (n_records,) — 각 시점의 현재 적합도 (탐색 궤적)
        hist_temp   : ndarray (n_records,) — 각 시점의 온도
        hist_accept : ndarray (n_records,) — 각 시점 직전 구간의 수용률
        n_eval      : int   — 총 목적함수 평가 횟수 (= 시도한 이동 수)
        n_accept    : int   — 수용된 이동 수
        n_uphill    : int   — 수용된 **악화** 이동 수 (지역최적 탈출의 증거)
        T0          : float — 사용된 초기 온도
        seconds     : float — 소요 시간

    Raises
    ------
    ValueError — 이진 문제가 아니거나 alpha 가 (0,1) 밖일 때

    Notes
    -----
    증분 평가(O(1)/이동)를 쓴다. 전체 재계산과 수학적으로 동일하며, 마지막에 `problem.fitness`
    로 검산해 부동소수점 누적오차가 없는지 확인한다.
    """
    if problem.n_crops != 2:
        raise ValueError('run_sa 는 이진(K=2) 문제만 지원한다. K=%d' % problem.n_crops)
    if not (0.0 < alpha < 1.0):
        raise ValueError('alpha 는 (0, 1) 이어야 한다. 받은 값: %r' % alpha)

    n = problem.n_units
    rng = np.random.default_rng(seed)

    adj = problem.adjusted_value(lam)          # (N,2) — 증분 평가의 재료
    area = problem.area

    # ── 초기 상태 ────────────────────────────────────────────────────────────
    z = (problem.baseline.copy() if init is None else np.asarray(init).astype(np.int8).copy())
    if feasible_state is not None and not feasible_state(z):
        raise ValueError('초기 상태가 실현 불가능하다. init 를 실현가능하게 주거나 repair 하라.')

    cur_fit = problem.fitness(z, lam)
    best_fit = cur_fit
    best = z.copy()

    # ── 초기 온도 ────────────────────────────────────────────────────────────
    if T0 is None:
        T0 = estimate_T0(problem, lam, z, np.random.default_rng(seed + 10_000),
                         target_accept=target_accept, n_flips=n_flips)
    T = float(T0)

    moves_per_level = max(1, n_iter // n_temp_levels)
    record_every = max(1, n_iter // n_records)

    hist_fit, hist_cur, hist_ev, hist_T, hist_acc = [], [], [], [], []
    n_accept = n_uphill = 0
    acc_window = 0                                   # 기록 구간 내 수용 수
    win_moves = 0
    it = 0
    t0 = time.perf_counter()

    # ── 냉각 루프 ────────────────────────────────────────────────────────────
    while it < n_iter:
        for _ in range(moves_per_level):
            if it >= n_iter:
                break

            # 이웃 제안
            idx = rng.choice(n, size=n_flips, replace=False)
            dF = _delta_fitness(adj, area, z, idx)   # 적합도 변화 (O(1))
            dE = -dF                                 # 에너지 변화

            # 제약이 있으면 실현 불가능한 이웃은 거부
            ok = True
            if feasible_state is not None:
                z[idx] = 1 - z[idx]
                ok = bool(feasible_state(z))
                z[idx] = 1 - z[idx]                  # 원복 (아래에서 수용 시 다시 뒤집는다)

            # Metropolis 수용 판정
            if ok and (dE <= 0.0 or rng.random() < np.exp(-dE / T)):
                z[idx] = 1 - z[idx]
                cur_fit += dF
                n_accept += 1
                acc_window += 1
                if dE > 0.0:
                    n_uphill += 1                    # 악화를 감수한 이동 = 탈출 시도
                if cur_fit > best_fit:
                    best_fit = cur_fit
                    best = z.copy()

            it += 1
            win_moves += 1

            if it % record_every == 0:
                hist_fit.append(best_fit)
                hist_cur.append(cur_fit)
                hist_ev.append(it)
                hist_T.append(T)
                hist_acc.append(acc_window / max(1, win_moves))
                acc_window = win_moves = 0

        T *= alpha                                   # 지수 냉각

    seconds = time.perf_counter() - t0

    # 증분 평가 검산 — 누적오차가 없는지 (전체 재계산과 대조)
    recomputed = problem.fitness(best, lam)
    if abs(recomputed - best_fit) > 1e-6 * max(1.0, abs(recomputed)):
        raise AssertionError('증분 평가 불일치: %.6f vs %.6f' % (best_fit, recomputed))
    best_fit = recomputed

    return dict(best=best.astype(np.int8), best_fit=float(best_fit),
                history=np.array(hist_fit), hist_evals=np.array(hist_ev),
                hist_cur=np.array(hist_cur), hist_temp=np.array(hist_T),
                hist_accept=np.array(hist_acc),
                n_eval=it, n_accept=n_accept, n_uphill=n_uphill,
                T0=float(T0), seconds=seconds)


# ══════════════════════════════════════════════════════════════════════════════
# λ 스윕 (SA 판)
# ══════════════════════════════════════════════════════════════════════════════
def sweep_lambda_sa(problem: CropAllocationProblem,
                    lambdas,
                    v_no_adapt: float,
                    loss: float,
                    **sa_kwargs) -> list:
    """전환비용 λ 를 스윕하며 SA 로 각 λ 의 해를 구한다.

    `optimize_ga.sweep_lambda` 와 같은 레코드 형식을 돌려주므로, 두 결과를 그대로 겹쳐
    트레이드오프 곡선을 비교할 수 있다.

    Parameters
    ----------
    problem    : CropAllocationProblem
    lambdas    : λ 값들
    v_no_adapt : float — 무조정 시 총가치
    loss       : float — 되찾아야 할 손실
    **sa_kwargs : run_sa 로 넘길 인자 (n_iter, alpha, seed 등)

    Returns
    -------
    list[dict] — lam, total_value, fitness, n_switch, pct_units, pct_area,
                 recovery_pct, seconds
    """
    rows = []
    for lam in lambdas:
        r = run_sa(problem, lam, **sa_kwargs)
        z = r['best']
        rows.append(dict(
            lam=float(lam),
            total_value=problem.total_value(z),
            fitness=r['best_fit'],
            n_switch=problem.n_switched(z),
            pct_units=100.0 * problem.n_switched(z) / problem.n_units,
            pct_area=100.0 * problem.switch_area(z) / problem.total_area,
            recovery_pct=problem.recovery_pct(z, v_no_adapt, loss),
            seconds=r['seconds'],
        ))
    return rows
