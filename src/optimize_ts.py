# -*- coding: utf-8 -*-
"""
단작 작물 배분 조합 최적화 — 타부 서치(Tabu Search, TS) 구현.

개요 v3 §5 Phase 5 의 **A→C 확장 3단계**. `optimize_ga.py`(GA)·`optimize_sa.py`(SA)와
**완전히 같은 문제**를 풀어 3방향으로 공정하게 비교하기 위한 모듈이다.
문제 정의·목적함수·전환비용은 `optimize_ga.CropAllocationProblem` 을 그대로 재사용한다
(재정의하지 않는다).

──────────────────────────────────────────────────────────────────────────────
TS 설계
──────────────────────────────────────────────────────────────────────────────
상태(state)   : 길이 N 의 이진 벡터 z. GA 의 염색체·SA 의 상태와 **같은 표현**.
이웃(neighbor): 카운티 하나의 작물을 뒤집기(1-flip). 이웃 전체 또는 후보 리스트에서 고른다.
이동(move)    : **최선 이동(best improvement)** — 이웃 중 적합도가 가장 높아지는 곳으로 간다.
                개선 이동이 없어도 **반드시 이동한다**(가장 덜 나쁜 곳으로). 이것이 TS 가
                지역 최적을 빠져나오는 방식이다 — SA 의 확률적 수용과 대비된다.
타부(tabu)    : 방금 뒤집은 카운티를 `tabu_tenure` 회 동안 다시 뒤집지 못하게 금지한다.
                → 직전 이동을 즉시 되돌리는 **순환(cycling)** 을 막는다.
                (구현: tabu_until[c] = it + tenure, it < tabu_until[c] 이면 금지)
열망(aspiration): 타부여도 그 이동이 **역대 최고 해를 갱신**한다면 금지를 푼다.
                  좋은 해를 타부 때문에 놓치는 것을 방지한다.

──────────────────────────────────────────────────────────────────────────────
증분 평가 — 이웃 전체를 O(N) 에 한 번에 평가한다
──────────────────────────────────────────────────────────────────────────────
목적함수가 카운티별로 분리되므로 (`optimize_sa` 와 같은 성질)

    ΔF(c) = A_c · ( ṽ[c, 1−z_c] − ṽ[c, z_c] ),   ṽ[c,k] = v[c,k] − λ·1[k ≠ x̄_c]

즉 **모든 이웃의 ΔF 를 벡터 연산 한 방에** 얻는다. TS 의 "이웃 전체 평가"가 비싸지 않은 이유다.

──────────────────────────────────────────────────────────────────────────────
평가 예산과 후보 리스트 — 공정 비교의 핵심
──────────────────────────────────────────────────────────────────────────────
TS 는 한 번 이동할 때마다 이웃 |Nbhd| 개를 **평가**한다. 이웃 전체를 쓰면 반복 1회 =
N(=2,142)회 평가이므로, GA·SA 와 같은 예산(160,200회)에서는 겨우 **74회**밖에 이동하지 못한다.
그런데 이 문제의 최적해는 현재 배분에서 238개 카운티를 뒤집어야 도달한다 → **예산 부족.**

그래서 **후보 리스트 전략(candidate list strategy)** 을 쓴다: 매 반복 무작위로 뽑은
`n_candidates` 개만 평가한다. 예산 B, 후보 k 라면 이동 횟수는 B/k 가 되어
k 를 줄일수록 더 많이 이동할 수 있다. 이는 TS 문헌의 표준 기법이며, 큰 이웃을 가진
문제에서 사실상 필수다.

  · `n_candidates=None` → 이웃 전체 (정석 TS, 예산을 많이 먹는다)
  · `n_candidates=k`    → 후보 리스트 (예산 맞추기에 유리)

──────────────────────────────────────────────────────────────────────────────
사용 예
──────────────────────────────────────────────────────────────────────────────
    from optimize_ga import CropAllocationProblem
    from optimize_ts import run_ts

    prob = CropAllocationProblem(value=V, area=A, baseline=XBAR)
    res  = run_ts(prob, lam=42.0, n_iter=1602, n_candidates=100,
                  tabu_tenure=20, seed=42)
    print(res['best_fit'], res['n_tabu_blocked'], res['n_aspiration'])
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from optimize_ga import CropAllocationProblem

__all__ = [
    'run_ts',
    'sweep_lambda_ts',
]


# ══════════════════════════════════════════════════════════════════════════════
# TS 메인 루프
# ══════════════════════════════════════════════════════════════════════════════
def run_ts(problem: CropAllocationProblem,
           lam: float,
           n_iter: int = 1602,
           n_candidates: int | None = 100,
           tabu_tenure: int = 20,
           seed: int = 42,
           init: np.ndarray | None = None,
           aspiration: bool = True,
           feasible_state: Callable[[np.ndarray], bool] | None = None,
           n_records: int = 800) -> dict:
    """타부 서치(TS)로 단작 배분 문제를 푼다.

    GA(`optimize_ga.run_ga`)·SA(`optimize_sa.run_sa`)와 **같은 문제·같은 목적함수**를 풀며,
    비교를 위해 반환 형식도 맞췄다.

    Parameters
    ----------
    problem        : CropAllocationProblem — K=2(이진) 문제
    lam            : float — 전환비용 세기 ($/ac)
    n_iter         : int   — 이동 횟수. 총 평가 횟수는 n_iter × (후보 수) 가 된다.
    n_candidates   : int | None
        매 반복 평가할 이웃의 수(후보 리스트 크기). None 이면 이웃 전체(N개)를 평가한다.
        GA·SA 와 평가 예산을 맞추려면 n_iter × n_candidates 를 그 예산에 맞춘다.
    tabu_tenure    : int   — 타부 기간. 뒤집은 카운티를 이 횟수만큼 다시 못 뒤집는다.
    seed           : int   — 난수 시드 (재현성)
    init           : ndarray (N,) — 초기 상태. None 이면 현재 배분 x̄ 에서 출발.
    aspiration     : bool  — 열망 기준 사용 여부. 타부여도 역대 최고를 갱신하면 허용한다.
    feasible_state : callable (N,) -> bool, optional
        제약 문제에서 상태의 실현가능성 판정. 실현 불가능해지는 이동은 건너뛰고
        차선 이동을 고른다.
    n_records      : int   — 수렴 곡선에 기록할 점의 개수

    Returns
    -------
    dict
        best            : ndarray (N,) int8 — 최고 상태
        best_fit        : float — 그 적합도 F_λ (GA·SA 와 부호를 맞췄다)
        history         : ndarray — 기록 시점별 best-so-far 적합도
        hist_evals      : ndarray — 각 기록 시점의 누적 평가 횟수
        hist_cur        : ndarray — 각 시점의 현재 적합도 (탐색 궤적)
        n_eval          : int — 총 평가 횟수 (= Σ 후보 수)
        n_iter          : int — 실제 수행한 이동 횟수
        n_uphill        : int — 적합도가 **나빠지는데도** 감행한 이동 수
                                (개선 이동이 없을 때 TS 가 지역최적을 탈출하는 방식)
        n_tabu_blocked  : int — 타부에 막힌 후보의 누적 수
        n_aspiration    : int — 열망 기준으로 타부를 푼 횟수
        seconds         : float — 소요 시간

    Raises
    ------
    ValueError — 이진 문제가 아니거나 파라미터가 범위를 벗어날 때

    Notes
    -----
    증분 평가(이웃 전체를 벡터 연산으로 O(k))를 쓴다. 마지막에 `problem.fitness` 로 검산해
    부동소수점 누적오차가 없는지 확인한다.
    """
    if problem.n_crops != 2:
        raise ValueError('run_ts 는 이진(K=2) 문제만 지원한다. K=%d' % problem.n_crops)
    if tabu_tenure < 0:
        raise ValueError('tabu_tenure 는 0 이상이어야 한다.')
    n = problem.n_units
    if n_candidates is not None and not (1 <= n_candidates <= n):
        raise ValueError('n_candidates 는 1..%d 또는 None 이어야 한다.' % n)

    rng = np.random.default_rng(seed)
    adj = problem.adjusted_value(lam)          # (N,2) — 증분 평가의 재료
    area = problem.area

    # ── 초기 상태 ────────────────────────────────────────────────────────────
    z = (problem.baseline.copy() if init is None else np.asarray(init).astype(np.int8).copy())
    if feasible_state is not None and not feasible_state(z):
        raise ValueError('초기 상태가 실현 불가능하다.')

    cur_fit = problem.fitness(z, lam)
    best_fit = cur_fit
    best = z.copy()

    tabu_until = np.zeros(n, dtype=np.int64)   # tabu_until[c] > it 이면 c 는 금지
    all_idx = np.arange(n)

    hist_fit, hist_cur, hist_ev = [], [], []
    n_eval = n_uphill = n_tabu_blocked = n_aspiration = 0
    record_every = max(1, n_iter // n_records)
    t0 = time.perf_counter()

    # ── 반복 ─────────────────────────────────────────────────────────────────
    it = 0
    for it in range(n_iter):
        # 후보 이웃
        cand = (all_idx if n_candidates is None
                else rng.choice(n, size=n_candidates, replace=False))

        # 이웃 전체의 ΔF 를 한 번에 (증분 평가)
        cur_crop = z[cand]
        new_crop = 1 - cur_crop
        delta = area[cand] * (adj[cand, new_crop] - adj[cand, cur_crop])
        n_eval += len(cand)

        # 타부 판정 + 열망 기준
        is_tabu = tabu_until[cand] > it
        n_tabu_blocked += int(is_tabu.sum())
        allowed = ~is_tabu
        if aspiration:
            # 타부여도 역대 최고를 갱신하면 허용
            aspir = is_tabu & (cur_fit + delta > best_fit + 1e-9)
            allowed = allowed | aspir

        if not allowed.any():
            # 후보가 전부 타부 → 이번 반복은 건너뛴다 (다음 반복에 새 후보를 뽑는다)
            if (it + 1) % record_every == 0:
                hist_fit.append(best_fit); hist_cur.append(cur_fit); hist_ev.append(n_eval)
            continue

        # 최선 이동(best improvement) — 실현 불가능하면 차선으로
        order = np.argsort(-delta)                       # ΔF 내림차순
        order = order[allowed[order]]                    # 허용된 것만
        chosen = -1
        for j in order:
            c = int(cand[j])
            if feasible_state is not None:
                z[c] = 1 - z[c]
                ok = bool(feasible_state(z))
                z[c] = 1 - z[c]
                if not ok:
                    continue
            chosen = j
            break
        if chosen < 0:
            if (it + 1) % record_every == 0:
                hist_fit.append(best_fit); hist_cur.append(cur_fit); hist_ev.append(n_eval)
            continue

        c = int(cand[chosen])
        dF = float(delta[chosen])
        was_tabu = bool(is_tabu[chosen])

        # 이동 감행 (개선이 아니어도 간다 — TS 의 탈출 메커니즘)
        z[c] = 1 - z[c]
        cur_fit += dF
        tabu_until[c] = it + 1 + tabu_tenure
        if dF < 0:
            n_uphill += 1
        if was_tabu:
            n_aspiration += 1
        if cur_fit > best_fit:
            best_fit = cur_fit
            best = z.copy()

        if (it + 1) % record_every == 0:
            hist_fit.append(best_fit); hist_cur.append(cur_fit); hist_ev.append(n_eval)

    seconds = time.perf_counter() - t0

    # 증분 평가 검산 — 전체 재계산과 대조
    recomputed = problem.fitness(best, lam)
    if abs(recomputed - best_fit) > 1e-6 * max(1.0, abs(recomputed)):
        raise AssertionError('증분 평가 불일치: %.6f vs %.6f' % (best_fit, recomputed))
    best_fit = recomputed

    return dict(best=best.astype(np.int8), best_fit=float(best_fit),
                history=np.array(hist_fit), hist_evals=np.array(hist_ev),
                hist_cur=np.array(hist_cur),
                n_eval=n_eval, n_iter=it + 1, n_uphill=n_uphill,
                n_tabu_blocked=n_tabu_blocked, n_aspiration=n_aspiration,
                seconds=seconds)


# ══════════════════════════════════════════════════════════════════════════════
# λ 스윕 (TS 판)
# ══════════════════════════════════════════════════════════════════════════════
def sweep_lambda_ts(problem: CropAllocationProblem,
                    lambdas,
                    v_no_adapt: float,
                    loss: float,
                    **ts_kwargs) -> list:
    """전환비용 λ 를 스윕하며 TS 로 각 λ 의 해를 구한다.

    `optimize_ga.sweep_lambda` · `optimize_sa.sweep_lambda_sa` 와 같은 레코드 형식을
    돌려주므로, 세 결과를 그대로 겹쳐 트레이드오프 곡선을 비교할 수 있다.

    Parameters
    ----------
    problem    : CropAllocationProblem
    lambdas    : λ 값들
    v_no_adapt : float — 무조정 시 총가치
    loss       : float — 되찾아야 할 손실
    **ts_kwargs : run_ts 로 넘길 인자 (n_iter, n_candidates, tabu_tenure, seed 등)

    Returns
    -------
    list[dict] — lam, total_value, fitness, n_switch, pct_units, pct_area,
                 recovery_pct, seconds
    """
    rows = []
    for lam in lambdas:
        r = run_ts(problem, lam, **ts_kwargs)
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
