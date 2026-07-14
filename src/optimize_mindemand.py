# -*- coding: utf-8 -*-
"""
최소 수요 제약이 있는 다작물 배분 최적화 — GA · SA · TS.

06~10의 문제는 카운티별로 **분리 가능**해서 정확해가 O(N·K) 에 나왔다. 그래서 메타휴리스틱은
"이미 아는 답을 맞히는" 검증 도구에 불과했다. 여기에 **최소 수요 제약**

    Σ_c  A_c · ŷ[c, corn] · 1[z_c = corn]  ≥  D_corn        (식량안보: 옥수수 총생산 하한)

를 걸면 카운티들이 **서로 묶인다.** 한 카운티에서 옥수수를 빼면 다른 카운티가 메꿔야 한다.
분리가능성이 깨지고, 문제는 **multiple-choice knapsack (MCKP)** 이 되어 NP-hard 다.
비로소 메타휴리스틱이 필요해진다.

──────────────────────────────────────────────────────────────────────────────
채점 기준 — 정확해가 없으니 상한이 필요하다
──────────────────────────────────────────────────────────────────────────────
정확해를 O(N·K) 에 구할 수 없으므로, **라그랑주 완화(Lagrangian relaxation)** 로
**증명 가능한 상한**을 만든다. 승수 μ ≥ 0 에 대해

    L(μ) = max_z [ Σ_c A_c·ṽ[c,z_c] + μ·(Σ_c coef[c]·1[z_c=corn] − D) ]
         = Σ_c max_k ( A_c·ṽ[c,k] + μ·coef[c]·1[k=corn] )  −  μ·D

는 **제약이 풀린** 문제라 다시 카운티별로 분리된다 → O(N·K) 에 계산된다.
약쌍대성(weak duality)에 의해 **모든 μ ≥ 0 에서 L(μ) ≥ (제약 있는 최적값)** 이다.
따라서 min_μ L(μ) 이 가장 좋은 상한이며, L(μ) 는 μ 의 볼록함수이므로 **이분 탐색**으로 찾는다.

이 상한으로 GA/SA/TS 의 **증명 가능한 최적성 갭**을 계산한다:

    gap = (L(μ*) − F_metaheuristic) / |L(μ*)|

(정수성 간극(integrality gap) 때문에 실제 최적값은 L(μ*) 보다 약간 낮을 수 있다.
즉 보고되는 갭은 **보수적**이다 — 진짜 갭은 이보다 작거나 같다.)

──────────────────────────────────────────────────────────────────────────────
제약 처리 — 페널티 + 수리(repair)
──────────────────────────────────────────────────────────────────────────────
메타휴리스틱의 적합도는 **페널티 방식**을 쓴다:

    F_pen(z) = F_λ(z) − ρ · max(0, D − production(z))

ρ 는 $/bu 단위이며, **ρ > μ\\*** 이면 페널티가 충분히 커서 최적해가 실현가능해진다
(라그랑주 이론). 그래서 μ* 를 먼저 구하고 **ρ = penalty_factor × μ\\*** 로 자동 설정한다.
탐색이 끝난 뒤에는 `repair()` 로 실현가능성을 **보장**한다.

**증분 평가**: 카운티 c 를 k_old → k_new 로 바꿀 때
  · 생산량 변화 = coef[c]·(1[k_new=corn] − 1[k_old=corn])
  · 목적함수 변화 = A_c·(ṽ[c,k_new] − ṽ[c,k_old])
둘 다 O(1) 이므로 페널티가 붙어도 SA/TS 의 증분 평가가 그대로 작동한다.

──────────────────────────────────────────────────────────────────────────────
사용 예
──────────────────────────────────────────────────────────────────────────────
    from optimize_ga import CropAllocationProblem
    from optimize_mindemand import MinDemandProblem, lagrangian_bound, run_ts_md

    base = CropAllocationProblem(value=V, area=A, baseline=XBAR)   # V: (N, 3)
    coef = A * y_hat_corn                       # 각 카운티가 옥수수를 고를 때의 생산량 (bu)
    mdp  = MinDemandProblem(base, crop=0, coef=coef, demand=D)

    ub, mu_star, _ = lagrangian_bound(mdp, lam=42.0)
    mdp.set_penalty_from_mu(mu_star, factor=3.0)
    res = run_ts_md(mdp, lam=42.0, seed=42)
    gap = 100 * (ub - res['best_fit']) / abs(ub)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from optimize_ga import CropAllocationProblem

__all__ = [
    'MinDemandProblem',
    'lagrangian_bound',
    'run_ga_md',
    'run_sa_md',
    'run_ts_md',
]


# ══════════════════════════════════════════════════════════════════════════════
# 제약 있는 문제
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class MinDemandProblem:
    """최소 수요 제약이 붙은 배분 문제.

    Parameters
    ----------
    base   : CropAllocationProblem — 제약 없는 기반 문제 (목적함수·정확해 제공)
    crop   : int — 하한을 거는 작물의 인덱스 (예: 옥수수 = 0)
    coef   : ndarray (N,) — 카운티 c 가 그 작물을 고를 때의 **생산량 기여**
             (= A_c · ŷ[c, crop], 단위 bu). 다른 작물을 고르면 기여 0.
    demand : float — 하한 D (bu)
    rho    : float — 페널티 계수 ($/bu). `set_penalty_from_mu` 로 자동 설정 권장.
    """

    base: CropAllocationProblem
    crop: int
    coef: np.ndarray
    demand: float
    rho: float = 0.0
    _idx: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.coef = np.asarray(self.coef, dtype=float)
        if self.coef.shape != (self.base.n_units,):
            raise ValueError('coef shape %r != (%d,)' % (self.coef.shape, self.base.n_units))
        if not (0 <= self.crop < self.base.n_crops):
            raise ValueError('crop 인덱스가 범위를 벗어난다.')
        if (self.coef < 0).any():
            raise ValueError('coef 는 음수일 수 없다.')
        self._idx = np.arange(self.base.n_units)

    # ── 크기 위임 ────────────────────────────────────────────────────────────
    @property
    def n_units(self) -> int:
        return self.base.n_units

    @property
    def n_crops(self) -> int:
        return self.base.n_crops

    @property
    def max_production(self) -> float:
        """모든 카운티가 그 작물을 골랐을 때의 총생산 (제약의 상한선)."""
        return float(self.coef.sum())

    # ── 제약 ─────────────────────────────────────────────────────────────────
    def production(self, z: np.ndarray) -> float:
        """배분 z 에서 제약 작물의 총생산 (bu)."""
        return float(self.coef[z == self.crop].sum())

    def production_pop(self, Z: np.ndarray) -> np.ndarray:
        """개체군 (P, N) 의 총생산 (P,)."""
        return ((Z == self.crop) * self.coef[None, :]).sum(axis=1)

    def shortfall(self, z: np.ndarray) -> float:
        """부족분 max(0, D − production). 0 이면 실현가능."""
        return max(0.0, self.demand - self.production(z))

    def is_feasible(self, z: np.ndarray, tol: float = 1e-6) -> bool:
        return self.production(z) >= self.demand - tol

    # ── 페널티 적합도 ────────────────────────────────────────────────────────
    def set_penalty_from_mu(self, mu: float, factor: float = 3.0) -> float:
        """라그랑주 승수 μ* 로부터 페널티 계수를 정한다: ρ = factor × μ*.

        ρ > μ* 이면 페널티가 충분히 커서 최적해가 실현가능해진다(라그랑주 이론).
        μ* 가 0 이면(제약 비구속) 작은 양수를 준다.

        Returns
        -------
        float — 설정된 ρ
        """
        self.rho = float(max(mu, 0.0) * factor) if mu > 0 else 1e-6
        return self.rho

    def penalized_fitness(self, z: np.ndarray, lam: float) -> float:
        """F_pen(z) = F_λ(z) − ρ · max(0, D − production(z))."""
        return self.base.fitness(z, lam) - self.rho * self.shortfall(z)

    def penalized_fitness_pop(self, Z: np.ndarray, lam: float) -> np.ndarray:
        """개체군 일괄 평가 (벡터화)."""
        f = self.base.fitness_pop(Z, lam)
        short = np.maximum(0.0, self.demand - self.production_pop(Z))
        return f - self.rho * short

    # ── 수리(repair) — 실현가능성 보장 ──────────────────────────────────────
    def repair(self, z: np.ndarray, lam: float) -> np.ndarray:
        """부족하면, **부셸당 목적함수 손실이 가장 싼 카운티부터** 제약 작물로 바꾼다.

        탐욕적 수리. 각 카운티를 제약 작물로 바꿀 때
          · 얻는 생산량 = coef[c]
          · 잃는 목적함수 = A_c·(ṽ[c, z_c] − ṽ[c, crop])   (≥ 0 일 수도, < 0 일 수도)
        비율(손실/부셸)이 작은 순으로 채운다.

        Returns
        -------
        ndarray (N,) int8 — 실현가능한 배분
        """
        z = np.asarray(z).astype(np.int8).copy()
        need = self.demand - self.production(z)
        if need <= 0:
            return z

        adj = self.base.adjusted_value(lam)          # (N, K)
        area = self.base.area
        cand = np.where(z != self.crop)[0]           # 바꿀 수 있는 카운티
        if len(cand) == 0:
            return z

        loss = area[cand] * (adj[cand, z[cand]] - adj[cand, self.crop])   # 목적함수 손실
        gain = self.coef[cand]                                            # 생산량 이득
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(gain > 0, loss / np.maximum(gain, 1e-12), np.inf)
        order = cand[np.argsort(ratio)]                                   # 싼 것부터

        for c in order:
            if need <= 0:
                break
            if self.coef[c] <= 0:
                continue
            z[c] = self.crop
            need -= self.coef[c]
        return z


# ══════════════════════════════════════════════════════════════════════════════
# 라그랑주 상한 — 메타휴리스틱을 채점할 기준
# ══════════════════════════════════════════════════════════════════════════════
def lagrangian_bound(mdp: MinDemandProblem, lam: float,
                     mu_hi: float | None = None, n_iter: int = 200,
                     tol: float = 1e-10):
    """라그랑주 완화로 **증명 가능한 상한**을 구한다 (이분 탐색).

    승수 μ ≥ 0 에 대해

        L(μ) = Σ_c max_k ( A_c·ṽ[c,k] + μ·coef[c]·1[k=crop] ) − μ·D

    는 제약을 목적함수로 흡수해 **다시 분리 가능**해진 문제이므로 O(N·K) 에 풀린다.
    약쌍대성에 의해 모든 μ ≥ 0 에서 L(μ) ≥ (제약 있는 최적값). L 은 μ 의 볼록함수이고
    부분기울기는 (production(z_μ) − D) 이므로, **생산량이 D 를 막 넘는 μ** 가 최소점이다.
    → 이분 탐색.

    Parameters
    ----------
    mdp    : MinDemandProblem
    lam    : float — 전환비용 세기
    mu_hi  : float — 이분 탐색 상한. None 이면 자동으로 넉넉히 잡는다.
    n_iter : int   — 이분 탐색 반복 수
    tol    : float

    Returns
    -------
    (bound, mu_star, z_mu) : (float, float, ndarray)
        bound  — min_μ L(μ). 제약 있는 최적값의 **상한**.
        mu_star— 그때의 승수 ($/bu). "제약을 1부셸 완화할 때의 목적함수 개선" 의 그림자가격.
        z_mu   — 그때의 완화해 (실현가능하지 않을 수 있다).

    Notes
    -----
    정수성 간극 때문에 실제 최적값은 bound 보다 낮을 수 있다. 따라서 이 bound 로 계산한
    최적성 갭은 **보수적**(실제 갭 ≤ 보고된 갭)이다.
    """
    base = mdp.base
    adj = base.adjusted_value(lam)                 # (N, K)
    area = base.area
    n, K = base.n_units, base.n_crops
    rows = np.arange(n)

    # 카운티별 목적 기여 (μ 없이): A_c · ṽ[c,k]
    W = area[:, None] * adj                        # (N, K)
    bonus = np.zeros((n, K))
    bonus[:, mdp.crop] = mdp.coef                  # μ 가 곱해질 항

    def solve(mu: float):
        M = W + mu * bonus
        z = M.argmax(axis=1).astype(np.int8)
        L = float(M[rows, z].sum() - mu * mdp.demand)
        prod = float(mdp.coef[z == mdp.crop].sum())
        return L, z, prod

    # 제약이 비구속이면 μ=0 이 최적
    L0, z0, prod0 = solve(0.0)
    if prod0 >= mdp.demand - 1e-6:
        return L0, 0.0, z0

    # μ 상한 자동 설정 — 생산량이 D 를 넘길 때까지 키운다
    if mu_hi is None:
        mu_hi = 1.0
        for _ in range(80):
            _, _, p = solve(mu_hi)
            if p >= mdp.demand:
                break
            mu_hi *= 2.0

    lo, hi = 0.0, float(mu_hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        _, _, p = solve(mid)
        if p >= mdp.demand:
            hi = mid          # 이미 충족 → μ 를 줄여도 된다
        else:
            lo = mid
        if hi - lo < tol * max(1.0, hi):
            break

    # 최소점 근처에서 양쪽을 모두 평가해 더 작은(=더 타이트한) 상한을 취한다
    cands = []
    for mu in (lo, hi, 0.5 * (lo + hi)):
        L, z, p = solve(mu)
        cands.append((L, mu, z))
    L_best, mu_best, z_best = min(cands, key=lambda t: t[0])
    return L_best, float(mu_best), z_best


# ══════════════════════════════════════════════════════════════════════════════
# 메타휴리스틱 — 페널티 + 수리
# ══════════════════════════════════════════════════════════════════════════════
def _random_reset(Z, rng, p_mut, K):
    """K작물 돌연변이 — 현재와 다른 작물로 무작위 교체."""
    mut = rng.random(Z.shape) < p_mut
    if not mut.any():
        return Z
    Z = Z.copy()
    cur = Z[mut]
    draw = rng.integers(0, K - 1, size=cur.shape[0])
    Z[mut] = (draw + (draw >= cur)).astype(np.int8)
    return Z


def run_ga_md(mdp: MinDemandProblem, lam: float,
              pop_size: int = 200, n_generations: int = 800,
              mutation_rate: float | None = None, crossover_rate: float = 0.9,
              tournament_k: int = 3, n_elite: int = 2, seed: int = 42,
              init: np.ndarray | None = None, repair_every: int = 0) -> dict:
    """제약 있는 문제용 GA (페널티 적합도 + 최종 수리).

    Parameters
    ----------
    mdp          : MinDemandProblem (rho 가 설정돼 있어야 한다)
    lam          : float
    repair_every : int — 0 이면 최종에만 수리. >0 이면 그 세대마다 개체군 전체를 수리한다.

    Returns
    -------
    dict — best(실현가능), best_fit(=제약 없는 F_λ), best_pen, history, n_eval,
           feasible, production, seconds
    """
    base = mdp.base
    n, K = base.n_units, base.n_crops
    rng = np.random.default_rng(seed)
    if mutation_rate is None:
        mutation_rate = 1.0 / n

    P = np.empty((pop_size, n), dtype=np.int8)
    half = pop_size // 2
    P[:half] = base.baseline[None, :]
    P[:half] = _random_reset(P[:half], rng, 0.05, K)
    P[half:] = rng.integers(0, K, size=(pop_size - half, n), dtype=np.int8)
    if init is not None:
        P[0] = init

    fit = mdp.penalized_fitness_pop(P, lam)
    history = np.empty(n_generations)
    n_eval = pop_size
    t0 = time.perf_counter()

    for g in range(n_generations):
        elite_idx = np.argsort(fit)[-n_elite:]
        elite, elite_fit = P[elite_idx].copy(), fit[elite_idx].copy()

        cand = rng.integers(0, pop_size, size=(pop_size, tournament_k))
        parents = P[cand[np.arange(pop_size), fit[cand].argmax(axis=1)]]
        p1, p2 = parents[0::2], parents[1::2]
        m = rng.random(p1.shape) < 0.5
        do_cx = rng.random((len(p1), 1)) < crossover_rate
        C = np.vstack([np.where(do_cx & m, p2, p1),
                       np.where(do_cx & m, p1, p2)])[:pop_size].astype(np.int8)
        C = _random_reset(C, rng, mutation_rate, K)

        if repair_every and (g + 1) % repair_every == 0:
            C = np.array([mdp.repair(C[i], lam) for i in range(pop_size)], dtype=np.int8)

        cf = mdp.penalized_fitness_pop(C, lam)
        n_eval += pop_size
        worst = np.argsort(cf)[:n_elite]
        C[worst], cf[worst] = elite, elite_fit
        P, fit = C, cf
        history[g] = fit.max()

    b = int(fit.argmax())
    z = mdp.repair(P[b], lam)                         # 최종 수리 — 실현가능성 보장
    return dict(best=z, best_fit=base.fitness(z, lam),
                best_pen=float(mdp.penalized_fitness(z, lam)),
                history=history, n_eval=n_eval,
                feasible=mdp.is_feasible(z), production=mdp.production(z),
                seconds=time.perf_counter() - t0)


def run_sa_md(mdp: MinDemandProblem, lam: float,
              n_iter: int = 160_200, n_temp_levels: int = 200, alpha: float = 0.95,
              T0: float | None = None, target_accept: float = 0.8,
              seed: int = 42, init: np.ndarray | None = None,
              n_records: int = 800) -> dict:
    """제약 있는 문제용 SA (페널티 + 증분 평가 + 최종 수리).

    증분 평가: 카운티 하나를 바꿀 때 목적함수 변화와 생산량 변화가 모두 O(1) 이므로,
    페널티가 붙어도 O(1) 로 ΔF_pen 을 구할 수 있다.
    """
    base = mdp.base
    n, K = base.n_units, base.n_crops
    rng = np.random.default_rng(seed)
    adj = base.adjusted_value(lam)
    area = base.area
    coef, crop, D, rho = mdp.coef, mdp.crop, mdp.demand, mdp.rho

    z = (base.baseline.copy() if init is None else np.asarray(init).astype(np.int8).copy())
    cur_f = base.fitness(z, lam)                       # 제약 없는 F
    cur_p = mdp.production(z)
    def pen(p):
        return rho * max(0.0, D - p)
    cur_pen = cur_f - pen(cur_p)
    best_pen, best = cur_pen, z.copy()

    def propose(rng_):
        c = int(rng_.integers(0, n))
        draw = int(rng_.integers(0, K - 1))
        return c, int(draw + (draw >= z[c]))

    def delta(c, new):
        dF = float(area[c] * (adj[c, new] - adj[c, z[c]]))
        # 주: numpy bool 끼리 빼면 TypeError — int 로 변환해야 한다
        dP = float(coef[c] * (int(new == crop) - int(z[c] == crop)))
        new_pen = (cur_f + dF) - pen(cur_p + dP)
        return dF, dP, new_pen - cur_pen

    if T0 is None:
        rp = np.random.default_rng(seed + 10_000)
        worse = []
        for _ in range(2000):
            c, new = propose(rp)
            _, _, dpen = delta(c, new)
            if -dpen > 0:
                worse.append(-dpen)
        T0 = float(np.mean(worse) / (-np.log(target_accept))) if worse else 1.0
    T = float(T0)

    moves_per_level = max(1, n_iter // n_temp_levels)
    record_every = max(1, n_iter // n_records)
    hist, hist_ev = [], []
    n_accept = n_uphill = 0
    it = 0
    t0 = time.perf_counter()

    while it < n_iter:
        for _ in range(moves_per_level):
            if it >= n_iter:
                break
            c, new = propose(rng)
            dF, dP, dpen = delta(c, new)
            dE = -dpen
            if dE <= 0.0 or rng.random() < np.exp(-dE / T):
                z[c] = new
                cur_f += dF
                cur_p += dP
                cur_pen += dpen
                n_accept += 1
                if dE > 0.0:
                    n_uphill += 1
                if cur_pen > best_pen:
                    best_pen, best = cur_pen, z.copy()
            it += 1
            if it % record_every == 0:
                hist.append(best_pen); hist_ev.append(it)
        T *= alpha

    seconds = time.perf_counter() - t0
    zf = mdp.repair(best, lam)
    return dict(best=zf, best_fit=base.fitness(zf, lam),
                best_pen=float(mdp.penalized_fitness(zf, lam)),
                history=np.array(hist), hist_evals=np.array(hist_ev),
                n_eval=it, n_accept=n_accept, n_uphill=n_uphill,
                feasible=mdp.is_feasible(zf), production=mdp.production(zf),
                T0=float(T0), seconds=seconds)


def run_ts_md(mdp: MinDemandProblem, lam: float,
              n_iter: int = 1000, tabu_tenure: int = 20, seed: int = 42,
              init: np.ndarray | None = None, aspiration: bool = True,
              n_records: int = 800) -> dict:
    """제약 있는 문제용 TS (이웃 전체 평가 + 페널티 + 최종 수리).

    이웃 전체의 ΔF_pen 을 (N, K) 행렬 연산으로 한 번에 구한다.
    페널티는 총생산의 함수이므로, 각 이웃이 만드는 생산량 변화도 (N, K) 로 벡터화된다.
    """
    base = mdp.base
    n, K = base.n_units, base.n_crops
    rng = np.random.default_rng(seed)
    adj = base.adjusted_value(lam)
    area = base.area
    coef, crop, D, rho = mdp.coef, mdp.crop, mdp.demand, mdp.rho
    rows = np.arange(n)

    is_crop = np.zeros((n, K))
    is_crop[:, crop] = 1.0                       # (N,K) — 그 작물이면 1

    z = (base.baseline.copy() if init is None else np.asarray(init).astype(np.int8).copy())
    cur_f = base.fitness(z, lam)
    cur_p = mdp.production(z)
    cur_pen = cur_f - rho * max(0.0, D - cur_p)
    best_pen, best = cur_pen, z.copy()

    tabu_until = np.zeros(n, dtype=np.int64)
    hist, hist_ev = [], []
    n_eval = n_uphill = n_tabu_blocked = 0
    record_every = max(1, n_iter // n_records)
    t0 = time.perf_counter()

    it = 0
    for it in range(n_iter):
        # 이웃 전체 — 목적함수 변화와 생산량 변화를 (N,K) 로
        dF = area[:, None] * (adj - adj[rows, z][:, None])           # (N,K)
        dP = coef[:, None] * (is_crop - is_crop[rows, z][:, None])   # (N,K)
        new_pen = (cur_f + dF) - rho * np.maximum(0.0, D - (cur_p + dP))
        dpen = new_pen - cur_pen
        dpen[rows, z] = -np.inf                                      # 자기 자신 제외
        n_eval += n * (K - 1)

        is_tabu = tabu_until > it
        n_tabu_blocked += int(is_tabu.sum()) * (K - 1)
        blocked = is_tabu.copy()
        if aspiration:
            best_gain = dpen.max(axis=1)
            aspir = is_tabu & (cur_pen + best_gain > best_pen + 1e-9)
            blocked = blocked & ~aspir

        d = dpen.copy()
        d[blocked, :] = -np.inf
        if not np.isfinite(d).any():
            continue

        flat = int(np.argmax(d))
        c, k = divmod(flat, K)
        z[c] = k
        cur_f += float(dF[c, k])
        cur_p += float(dP[c, k])
        step = float(dpen[c, k])
        cur_pen += step
        tabu_until[c] = it + 1 + tabu_tenure
        if step < 0:
            n_uphill += 1
        if cur_pen > best_pen:
            best_pen, best = cur_pen, z.copy()

        if (it + 1) % record_every == 0:
            hist.append(best_pen); hist_ev.append(n_eval)

    seconds = time.perf_counter() - t0
    zf = mdp.repair(best, lam)
    return dict(best=zf, best_fit=base.fitness(zf, lam),
                best_pen=float(mdp.penalized_fitness(zf, lam)),
                history=np.array(hist), hist_evals=np.array(hist_ev),
                n_eval=n_eval, n_iter=it + 1, n_uphill=n_uphill,
                n_tabu_blocked=n_tabu_blocked,
                feasible=mdp.is_feasible(zf), production=mdp.production(zf),
                seconds=seconds)
