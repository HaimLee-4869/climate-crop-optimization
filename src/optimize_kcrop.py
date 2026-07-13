# -*- coding: utf-8 -*-
"""
다작물(K ≥ 2) 단작 배분 조합 최적화 — GA · SA · TS.

06~09는 **2작물**(옥수수/대두) 문제를 풀었고, `optimize_ga.run_ga` · `optimize_sa.run_sa` ·
`optimize_ts.run_ts` 는 모두 **이진 전용**이다(bit-flip / 1-flip 이웃이 `1 - z` 를 쓴다).
밀을 더해 K=3 이 되면 그 연산자들을 쓸 수 없다.

이 모듈은 **K개 작물**로 일반화한 세 메타휴리스틱을 담는다. 문제 정의는
`optimize_ga.CropAllocationProblem` 을 **그대로 재사용**한다 — 그 클래스의 목적함수·정확해는
이미 K ≥ 2 를 지원한다(`fitness_pop`, `adjusted_value`, `exact_optimum`).

기존 06~09가 쓰는 `optimize_ga/sa/ts` 는 **건드리지 않는다.**

──────────────────────────────────────────────────────────────────────────────
2작물과 무엇이 달라지는가
──────────────────────────────────────────────────────────────────────────────
* **돌연변이 / 이웃**: `1 - z` (뒤집기)가 성립하지 않는다.
  → **random-reset**: 유전자를 *현재 값이 아닌* 다른 작물 중 하나로 무작위 교체.
* **TS 이웃**: 카운티당 이웃이 1개가 아니라 **K−1개**(다른 모든 작물)다.
  → 이웃 크기가 N·(K−1) 로 커진다. 이웃 전체의 ΔF 는 여전히 벡터 연산으로 O(N·K).
* **타부**: "카운티 c 를 건드림"을 금지하는 방식(속성 기반)을 유지한다.

목적함수의 **분리가능성은 K에 무관하게 성립**한다:

    F_λ(z) = Σ_c A_c · ṽ[c, z_c],   ṽ[c,k] = v[c,k] − λ·1[k ≠ x̄_c]

따라서 `problem.exact_optimum(lam)` 은 K작물에서도 O(N·K) 에 **정확 최적해**를 준다.
메타휴리스틱은 그 정확해로 채점된다.

──────────────────────────────────────────────────────────────────────────────
사용 예
──────────────────────────────────────────────────────────────────────────────
    from optimize_ga import CropAllocationProblem
    from optimize_kcrop import run_ga_k, run_sa_k, run_ts_k, random_baseline

    prob = CropAllocationProblem(value=V, area=A, baseline=XBAR)   # V: (N, 3)
    res = run_ga_k(prob, lam=42.0, seed=42)
    print(res['best_fit'], prob.exact_optimum(42.0))
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from optimize_ga import CropAllocationProblem

__all__ = [
    'run_ga_k',
    'run_sa_k',
    'run_ts_k',
    'greedy_k',
    'random_baseline',
    'sweep_lambda_k',
]


# ══════════════════════════════════════════════════════════════════════════════
# 공통 도구
# ══════════════════════════════════════════════════════════════════════════════
def greedy_k(problem: CropAllocationProblem) -> np.ndarray:
    """전환비용을 무시하고 카운티마다 가치가 최대인 작물을 고른다 (= λ=0 정확해).

    Returns
    -------
    ndarray (N,) int8
    """
    return problem.exact_optimum(lam=0.0)


def random_baseline(problem: CropAllocationProblem, lam: float, n: int = 2000,
                    seed: int = 42) -> dict:
    """무작위 배분 n개를 생성해 목적함수 분포를 만든다 (sanity check용).

    Parameters
    ----------
    problem : CropAllocationProblem
    lam     : float
    n       : int — 생성할 무작위 배분 수
    seed    : int

    Returns
    -------
    dict — fitness (n,), mean, std, max, min
    """
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, problem.n_crops, size=(n, problem.n_units), dtype=np.int8)
    f = problem.fitness_pop(Z, lam)
    return dict(fitness=f, mean=float(f.mean()), std=float(f.std()),
                max=float(f.max()), min=float(f.min()))


def _random_reset(Z: np.ndarray, rng: np.random.Generator, p_mut: float,
                  n_crops: int) -> np.ndarray:
    """K작물용 돌연변이 — 유전자를 **현재 값이 아닌** 다른 작물로 무작위 교체.

    이진 문제의 bit-flip(`1 - z`)을 K작물로 일반화한 것이다.
    현재 값을 다시 뽑는 낭비를 막기 위해, [0, K-2] 에서 뽑은 뒤 현재 값 이상이면 +1 한다.

    Parameters
    ----------
    Z       : ndarray (P, N)
    rng     : np.random.Generator
    p_mut   : float — 유전자당 변이 확률
    n_crops : int   — K

    Returns
    -------
    ndarray (P, N) int8
    """
    mut = rng.random(Z.shape) < p_mut
    if not mut.any():
        return Z
    Z = Z.copy()
    cur = Z[mut]
    draw = rng.integers(0, n_crops - 1, size=cur.shape[0])      # K-1 개 중 하나
    new = draw + (draw >= cur)                                   # 현재 값을 건너뛴다
    Z[mut] = new.astype(np.int8)
    return Z


# ══════════════════════════════════════════════════════════════════════════════
# GA (K작물)
# ══════════════════════════════════════════════════════════════════════════════
def run_ga_k(problem: CropAllocationProblem,
             lam: float,
             pop_size: int = 200,
             n_generations: int = 800,
             mutation_rate: float | None = None,
             crossover_rate: float = 0.9,
             tournament_k: int = 3,
             n_elite: int = 2,
             seed: int = 42,
             init: np.ndarray | None = None,
             init_jitter: float = 0.05) -> dict:
    """K작물 유전 알고리즘.

    `optimize_ga.run_ga` 와 동일한 구조이되, 돌연변이만 bit-flip → **random-reset** 으로 바꿨다.
    선택(토너먼트)·교차(균등)·엘리트 보존은 그대로다.

    Parameters
    ----------
    problem        : CropAllocationProblem (K ≥ 2)
    lam            : float — 전환비용 세기 ($/ac)
    pop_size       : int
    n_generations  : int
    mutation_rate  : float — None 이면 1/N
    crossover_rate : float
    tournament_k   : int
    n_elite        : int
    seed           : int
    init           : ndarray (N,) — 0번 개체로 주입할 시드 해
    init_jitter    : float — 초기화 시 x̄ 를 흔드는 정도

    Returns
    -------
    dict — best, best_fit, history, n_eval, seconds
    """
    n, K = problem.n_units, problem.n_crops
    rng = np.random.default_rng(seed)
    if mutation_rate is None:
        mutation_rate = 1.0 / n

    # 초기 개체군: 절반은 x̄ 를 흔든 것, 절반은 무작위
    P = np.empty((pop_size, n), dtype=np.int8)
    half = pop_size // 2
    P[:half] = problem.baseline[None, :]
    P[:half] = _random_reset(P[:half], rng, init_jitter, K)
    P[half:] = rng.integers(0, K, size=(pop_size - half, n), dtype=np.int8)
    if init is not None:
        P[0] = init

    fit = problem.fitness_pop(P, lam)
    history = np.empty(n_generations)
    n_eval = pop_size
    t0 = time.perf_counter()

    for g in range(n_generations):
        elite_idx = np.argsort(fit)[-n_elite:]
        elite, elite_fit = P[elite_idx].copy(), fit[elite_idx].copy()

        # 토너먼트 선택
        cand = rng.integers(0, pop_size, size=(pop_size, tournament_k))
        parents = P[cand[np.arange(pop_size), fit[cand].argmax(axis=1)]]

        # 균등 교차
        p1, p2 = parents[0::2], parents[1::2]
        m = rng.random(p1.shape) < 0.5
        do_cx = rng.random((len(p1), 1)) < crossover_rate
        C = np.vstack([np.where(do_cx & m, p2, p1),
                       np.where(do_cx & m, p1, p2)])[:pop_size].astype(np.int8)

        # random-reset 돌연변이 (K작물)
        C = _random_reset(C, rng, mutation_rate, K)

        cf = problem.fitness_pop(C, lam)
        n_eval += pop_size

        worst = np.argsort(cf)[:n_elite]
        C[worst], cf[worst] = elite, elite_fit
        P, fit = C, cf
        history[g] = fit.max()

    b = int(fit.argmax())
    return dict(best=P[b].copy().astype(np.int8), best_fit=float(fit[b]),
                history=history, n_eval=n_eval,
                seconds=time.perf_counter() - t0)


# ══════════════════════════════════════════════════════════════════════════════
# SA (K작물)
# ══════════════════════════════════════════════════════════════════════════════
def run_sa_k(problem: CropAllocationProblem,
             lam: float,
             n_iter: int = 160_200,
             n_temp_levels: int = 200,
             alpha: float = 0.95,
             T0: float | None = None,
             target_accept: float = 0.8,
             seed: int = 42,
             init: np.ndarray | None = None,
             n_records: int = 800) -> dict:
    """K작물 담금질 기법.

    이웃 = 무작위 카운티 하나를 **다른 작물 중 하나로** 무작위 교체(random-reset).
    증분 평가: ΔF = A_c · (ṽ[c, new] − ṽ[c, cur]) — O(1). K에 무관하다.

    Parameters
    ----------
    problem       : CropAllocationProblem (K ≥ 2)
    lam           : float
    n_iter        : int   — 총 이동 시도 횟수 (= 평가 횟수)
    n_temp_levels : int   — 온도 레벨 수
    alpha         : float — 지수 냉각 계수
    T0            : float — 초기 온도. None 이면 목표 수용확률에서 자동 산출.
    target_accept : float — T0 자동 산출 시의 목표 초기 수용확률
    seed          : int
    init          : ndarray (N,) — 초기 상태. None 이면 x̄.
    n_records     : int

    Returns
    -------
    dict — best, best_fit, history, hist_evals, hist_cur, hist_temp,
           n_eval, n_accept, n_uphill, T0, seconds
    """
    n, K = problem.n_units, problem.n_crops
    if not (0.0 < alpha < 1.0):
        raise ValueError('alpha 는 (0,1) 이어야 한다.')
    rng = np.random.default_rng(seed)

    adj = problem.adjusted_value(lam)          # (N, K)
    area = problem.area

    z = (problem.baseline.copy() if init is None
         else np.asarray(init).astype(np.int8).copy())
    cur_fit = problem.fitness(z, lam)
    best_fit, best = cur_fit, z.copy()

    def propose(rng_):
        """무작위 카운티 + 현재와 다른 작물."""
        c = int(rng_.integers(0, n))
        draw = int(rng_.integers(0, K - 1))
        new = draw + (draw >= z[c])
        return c, int(new)

    # ── 초기 온도: 악화 이동의 평균 크기에서 역산 ────────────────────────────
    if T0 is None:
        rp = np.random.default_rng(seed + 10_000)
        worse = []
        for _ in range(2000):
            c, new = propose(rp)
            dE = -(area[c] * (adj[c, new] - adj[c, z[c]]))
            if dE > 0:
                worse.append(dE)
        T0 = float(np.mean(worse) / (-np.log(target_accept))) if worse else 1.0
    T = float(T0)

    moves_per_level = max(1, n_iter // n_temp_levels)
    record_every = max(1, n_iter // n_records)
    hist_fit, hist_cur, hist_ev, hist_T = [], [], [], []
    n_accept = n_uphill = 0
    it = 0
    t0 = time.perf_counter()

    while it < n_iter:
        for _ in range(moves_per_level):
            if it >= n_iter:
                break
            c, new = propose(rng)
            dF = float(area[c] * (adj[c, new] - adj[c, z[c]]))   # O(1) 증분 평가
            dE = -dF
            if dE <= 0.0 or rng.random() < np.exp(-dE / T):
                z[c] = new
                cur_fit += dF
                n_accept += 1
                if dE > 0.0:
                    n_uphill += 1
                if cur_fit > best_fit:
                    best_fit, best = cur_fit, z.copy()
            it += 1
            if it % record_every == 0:
                hist_fit.append(best_fit); hist_cur.append(cur_fit)
                hist_ev.append(it); hist_T.append(T)
        T *= alpha

    seconds = time.perf_counter() - t0
    recomputed = problem.fitness(best, lam)          # 증분 평가 검산
    if abs(recomputed - best_fit) > 1e-6 * max(1.0, abs(recomputed)):
        raise AssertionError('증분 평가 불일치: %.6f vs %.6f' % (best_fit, recomputed))

    return dict(best=best.astype(np.int8), best_fit=float(recomputed),
                history=np.array(hist_fit), hist_evals=np.array(hist_ev),
                hist_cur=np.array(hist_cur), hist_temp=np.array(hist_T),
                n_eval=it, n_accept=n_accept, n_uphill=n_uphill,
                T0=float(T0), seconds=seconds)


# ══════════════════════════════════════════════════════════════════════════════
# TS (K작물)
# ══════════════════════════════════════════════════════════════════════════════
def run_ts_k(problem: CropAllocationProblem,
             lam: float,
             n_iter: int = 1000,
             tabu_tenure: int = 20,
             seed: int = 42,
             init: np.ndarray | None = None,
             aspiration: bool = True,
             n_records: int = 800) -> dict:
    """K작물 타부 서치 (이웃 전체 평가).

    이웃 = (카운티 c, 작물 k≠z_c) 인 모든 쌍 → **N·(K−1)개**. 2작물의 N개보다 커진다.
    이웃 전체의 ΔF 를 (N, K) 행렬 연산으로 한 번에 구한다:

        ΔF[c, k] = A_c · (ṽ[c,k] − ṽ[c, z_c])

    타부는 **카운티 단위**로 건다(방금 건드린 카운티를 tenure 회 동안 재금지).

    Parameters
    ----------
    problem     : CropAllocationProblem (K ≥ 2)
    lam         : float
    n_iter      : int — 이동 횟수. 평가 횟수는 n_iter × N × (K−1).
    tabu_tenure : int
    seed        : int
    init        : ndarray (N,)
    aspiration  : bool — 타부여도 역대 최고 갱신 시 허용
    n_records   : int

    Returns
    -------
    dict — best, best_fit, history, hist_evals, hist_cur,
           n_eval, n_iter, n_uphill, n_tabu_blocked, n_aspiration, seconds
    """
    n, K = problem.n_units, problem.n_crops
    rng = np.random.default_rng(seed)
    adj = problem.adjusted_value(lam)          # (N, K)
    area = problem.area
    rows = np.arange(n)

    z = (problem.baseline.copy() if init is None
         else np.asarray(init).astype(np.int8).copy())
    cur_fit = problem.fitness(z, lam)
    best_fit, best = cur_fit, z.copy()

    tabu_until = np.zeros(n, dtype=np.int64)
    hist_fit, hist_cur, hist_ev = [], [], []
    n_eval = n_uphill = n_tabu_blocked = n_aspiration = 0
    record_every = max(1, n_iter // n_records)
    t0 = time.perf_counter()

    it = 0
    for it in range(n_iter):
        # 이웃 전체의 ΔF — (N, K) 한 방에
        cur_val = adj[rows, z]                                  # (N,)
        delta = area[:, None] * (adj - cur_val[:, None])        # (N, K)
        delta[rows, z] = -np.inf                                # 자기 자신은 이웃 아님
        n_eval += n * (K - 1)

        # 카운티 단위 타부
        is_tabu = tabu_until > it
        n_tabu_blocked += int(is_tabu.sum()) * (K - 1)
        blocked = is_tabu.copy()
        if aspiration:
            best_gain = delta.max(axis=1)                       # 카운티별 최선 이동
            aspir = is_tabu & (cur_fit + best_gain > best_fit + 1e-9)
            blocked = blocked & ~aspir
            n_aspiration += int(aspir.sum())

        d = delta.copy()
        d[blocked, :] = -np.inf
        if not np.isfinite(d).any():
            if (it + 1) % record_every == 0:
                hist_fit.append(best_fit); hist_cur.append(cur_fit); hist_ev.append(n_eval)
            continue

        flat = int(np.argmax(d))
        c, k = divmod(flat, K)
        dF = float(delta[c, k])

        z[c] = k                                                # 이동 감행 (악화여도)
        cur_fit += dF
        tabu_until[c] = it + 1 + tabu_tenure
        if dF < 0:
            n_uphill += 1
        if cur_fit > best_fit:
            best_fit, best = cur_fit, z.copy()

        if (it + 1) % record_every == 0:
            hist_fit.append(best_fit); hist_cur.append(cur_fit); hist_ev.append(n_eval)

    seconds = time.perf_counter() - t0
    recomputed = problem.fitness(best, lam)
    if abs(recomputed - best_fit) > 1e-6 * max(1.0, abs(recomputed)):
        raise AssertionError('증분 평가 불일치: %.6f vs %.6f' % (best_fit, recomputed))

    return dict(best=best.astype(np.int8), best_fit=float(recomputed),
                history=np.array(hist_fit), hist_evals=np.array(hist_ev),
                hist_cur=np.array(hist_cur),
                n_eval=n_eval, n_iter=it + 1, n_uphill=n_uphill,
                n_tabu_blocked=n_tabu_blocked, n_aspiration=n_aspiration,
                seconds=seconds)


# ══════════════════════════════════════════════════════════════════════════════
# λ 스윕
# ══════════════════════════════════════════════════════════════════════════════
def sweep_lambda_k(problem: CropAllocationProblem,
                   lambdas,
                   v_no_adapt: float,
                   loss: float,
                   method: str = 'exact',
                   **kwargs) -> list:
    """λ 를 스윕하며 각 λ 의 해와 특성을 기록한다 (K작물).

    Parameters
    ----------
    problem    : CropAllocationProblem
    lambdas    : λ 값들
    v_no_adapt : float — 무조정 시 총가치
    loss       : float — 되찾아야 할 손실
    method     : 'exact' | 'ga' | 'sa' | 'ts'
    **kwargs   : 해당 메서드로 넘길 인자

    Returns
    -------
    list[dict] — lam, total_value, fitness, n_switch, pct_units, pct_area,
                 recovery_pct, crop_shares (작물별 면적 비율), seconds
    """
    runner = {'exact': None, 'ga': run_ga_k, 'sa': run_sa_k, 'ts': run_ts_k}[method]
    rows = []
    for lam in lambdas:
        t0 = time.perf_counter()
        if runner is None:
            z = problem.exact_optimum(lam)
            sec = time.perf_counter() - t0
        else:
            r = runner(problem, lam, **kwargs)
            z, sec = r['best'], r['seconds']
        shares = [100.0 * problem.crop_area(z, k) / problem.total_area
                  for k in range(problem.n_crops)]
        rows.append(dict(
            lam=float(lam), method=method,
            total_value=problem.total_value(z),
            fitness=problem.fitness(z, lam),
            n_switch=problem.n_switched(z),
            pct_units=100.0 * problem.n_switched(z) / problem.n_units,
            pct_area=100.0 * problem.switch_area(z) / problem.total_area,
            recovery_pct=problem.recovery_pct(z, v_no_adapt, loss),
            crop_shares=shares,
            seconds=sec,
        ))
    return rows
