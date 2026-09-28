"""메타 시스템: 월말 국면 → 같은 국면 뒤 과거 성과로 전략 선택 → 다음 달 운용 (워크포워드).

규칙(사전 고정, 계획 파일 참고)
- 표본: "국면 L 이 관측된 달의 다음 달" 전략 일간수익률(그 시점 이전 것만). 12개월 미만이면 전체 과거로 판정.
- 통과: 샤프 ≥ 0.5, 연수익률 > 0, MDD ≥ -30%. 통과 전략 중 샤프 상위 5, 슬롯당 20%, 빈 슬롯은 단기채(153130).
- 전략 이력 12개월 미만은 후보 제외. 전략 간 비중 변경분에 편도 0.1% 비용.
- 모드: none(국면 없음, 기준선 c), d1~d4(regimes.py).

    python meta.py            # 전 모드 워크포워드 + 유의성 검사 → results/meta_*.parquet/csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import kr_data as K
import regimes as RG

RES = Path(__file__).parent / "results"
CASH = "153130"
EXCLUDE = {"ON05",                                              # 상한가 체결 불가
           "CD01", "ON07", "SW14", "SW15", "SW16", "LT14",      # 수급 데이터 결함(재수집 전)
           "BM02"}                                          # 벤치마크(후보 아님)
SHARPE_MIN, CAGR_MIN, MDD_MIN, SLOTS, SLOT_W, MIN_MONTHS = 0.5, 0.0, -0.30, 5, 0.2, 12
SWITCH_COST = 0.001


# ─── 입력 ───
def load_inputs():
    d = pd.read_parquet(RES / "daily.parquet")
    d = d.drop(columns=[c for c in d.columns if c in EXCLUDE])
    cash = K.load("etf")[CASH]["Close"].dropna().pct_change()
    idx = d.index
    cash = cash.reindex(idx).fillna(0.0)
    months = idx.to_period("M")
    return d, cash, months


RF_KEY = "__rf__"


def month_blocks(d, months):
    """전략별 월 블록 (월 → 일간수익률 ndarray). 데이터 없는 달은 제외. RF_KEY 에 그달 단기채 수익률."""
    import engine as E
    rf = E.rf_daily(d.index)
    blocks = {}
    for m, g in d.groupby(months):
        blocks[m] = {c: g[c].to_numpy() for c in d.columns if g[c].notna().all() and len(g) > 0}
        blocks[m][RF_KEY] = rf.loc[g.index].to_numpy()
    return blocks


# ─── 지표 ───
def stats(r: np.ndarray, rf: np.ndarray = None):
    if len(r) < 20 or r.std() == 0:
        return None
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (252 / len(r)) - 1
    ex = r - rf if rf is not None else r            # 2026-09-25: 단기채 대비 초과 샤프
    sharpe = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else -np.inf
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return sharpe, cagr, mdd


def select(t, labels, blocks, all_months, strategies, window=None, rank_window=None, rank_by="sharpe"):
    """결정월 t(월말)에 쓸 전략과 근거. labels: 월→국면(없으면 None = 국면 없음 모드).
    window: 채점을 최근 N개월로만(B). rank_window: 전체 과거로 합격 판정 후 최근 N개월 샤프로 순위(C)."""
    past = [m for m in all_months if m <= t]                    # t 까지 끝난 달
    # 국면 L 이 관측된 달 m 의 '다음 달' m+1 (m+1 ≤ t)
    if labels is not None and t in labels.index and pd.notna(labels[t]):
        cur = labels[t]
        sample = [m + 1 for m in past if (m + 1) <= t and m in labels.index and labels[m] == cur]
        if len(sample) < MIN_MONTHS:
            sample = past
    else:
        sample = past[-window:] if window else past
    picks = []
    for s in strategies:
        hist_months = [m for m in past if s in blocks.get(m, {})]
        if len(hist_months) < MIN_MONTHS:
            continue
        ms_ = [m for m in sample if s in blocks.get(m, {})]
        if not ms_:
            continue
        st = stats(np.concatenate([blocks[m][s] for m in ms_]), np.concatenate([blocks[m][RF_KEY] for m in ms_]))
        if st and st[0] >= SHARPE_MIN and st[1] > CAGR_MIN and st[2] >= MDD_MIN:
            if rank_window:
                rw = [m for m in past[-rank_window:] if s in blocks.get(m, {})]
                rec = stats(np.concatenate([blocks[m][s] for m in rw]), np.concatenate([blocks[m][RF_KEY] for m in rw]))
                st = (rec[0] if rec else -np.inf, *st[1:])
            picks.append((s, *st))
    picks.sort(key=lambda x: -x[1])
    if rank_by != "sharpe" and picks:
        sh, cg, dd = (pd.Series([p[k] for p in picks]) for k in (1, 2, 3))
        key = {"composite": (sh.rank(pct=True) + cg.rank(pct=True) + dd.rank(pct=True)) / 3,
               "calmar": cg / dd.abs().clip(lower=1e-4), "cagr": cg}[rank_by]
        picks = [picks[i] for i in key.sort_values(ascending=False, kind="stable").index]
    return picks[:SLOTS], len(sample)


WINDOW_MODES = {"w12": dict(window=12), "w36": dict(window=36), "w60": dict(window=60), "mix12": dict(rank_window=12)}
RANK_MODES = {"r_composite": dict(rank_by="composite"), "r_calmar": dict(rank_by="calmar"), "r_cagr": dict(rank_by="cagr")}


def walk_forward(mode, d, cash, months, blocks, labels_df):
    kw = WINDOW_MODES.get(mode) or RANK_MODES.get(mode) or {}
    labels = None if mode == "none" or kw else labels_df[mode]
    all_months = sorted(blocks)
    strategies = list(d.columns)
    rets, log, prev_w = [], [], {}
    for i, t in enumerate(all_months[:-1]):
        nxt = all_months[i + 1]
        picks, n = select(t, labels, blocks, all_months, strategies, **kw)
        w = {p[0]: SLOT_W for p in picks}
        w_cash = 1 - SLOT_W * len(picks)
        g = d.loc[months == nxt]
        r = g[list(w)].fillna(0).to_numpy() @ np.array(list(w.values())) if w else np.zeros(len(g))
        r = r + w_cash * cash.loc[g.index].to_numpy()
        turn = sum(abs(w.get(k, 0) - prev_w.get(k, 0)) for k in set(w) | set(prev_w))
        r[0] -= SWITCH_COST * turn
        prev_w = w
        rets.append(pd.Series(r, index=g.index))
        log.append({"decision": str(t), "hold": str(nxt), "regime": (labels.get(t) if labels is not None else None),
                    "sample_months": n, "picks": ",".join(p[0] for p in picks),
                    "sharpes": ",".join(f"{p[1]:.2f}" for p in picks), "cash_w": round(w_cash, 2)})
    return pd.concat(rets), pd.DataFrame(log)


def baselines(d, cash, months, blocks):
    k = K.load("etf")["069500"]["Close"].dropna().pct_change().reindex(d.index)
    all_months = sorted(blocks)
    ew = []
    for i, t in enumerate(all_months[:-1]):
        nxt = all_months[i + 1]
        elig = [s for s in d.columns if sum(1 for m in all_months if m <= t and s in blocks[m]) >= MIN_MONTHS]
        g = d.loc[months == nxt, elig]
        ew.append(g.mean(axis=1) if elig else pd.Series(0.0, index=g.index))
    return k, pd.concat(ew)


# ─── 국면 유의성 사전 검사 ───
def next_month_returns(d, months):
    """전략별 월수익률(월 m 의 수익률)."""
    return (1 + d).groupby(months).prod(min_count=1) - 1


def regime_tests(labels: pd.Series, mret: pd.DataFrame, n_shift=None):
    """(차이) Kruskal-Wallis p<0.05 비율, (지속성) 에피소드 간 순위 상관 — 시간축 순환이동 대조군과 비교."""
    from scipy.stats import kruskal, spearmanr
    lab = labels.dropna()
    # 국면 m 뒤 m+1 수익률
    y = mret.reindex([m + 1 for m in lab.index]).set_axis(lab.index)

    def diff_share(lb):
        ps = []
        for c in y.columns:
            grp = [y[c][lb == k].dropna().to_numpy() for k in lb.unique()]
            grp = [g for g in grp if len(g) >= 6]
            if len(grp) >= 2:
                ps.append(kruskal(*grp).pvalue)
        return np.mean(np.array(ps) < 0.05) if ps else np.nan

    def persistence(lb):
        run = (lb != lb.shift()).cumsum()
        cors = []
        for k in lb.unique():
            eps = [g.index for _, g in lb[lb == k].groupby(run[lb == k])]
            for j in range(1, len(eps)):
                prev = y.loc[np.concatenate([e for e in eps[:j]])]
                now = y.loc[eps[j]]
                if len(now) < 3 or len(prev) < 6:
                    continue
                sp = prev.mean() / prev.std()
                sn = now.mean() / now.std()
                ok = sp.notna() & sn.notna()
                if ok.sum() >= 10:
                    cors.append(spearmanr(sp[ok], sn[ok]).statistic)
        return np.nanmean(cors) if cors else np.nan

    obs_d, obs_p = diff_share(lab), persistence(lab)
    vals = lab.to_numpy()
    shifts = range(6, len(vals) - 6) if n_shift is None else range(6, 6 + n_shift)
    null_d, null_p = [], []
    for s in shifts:
        lb = pd.Series(np.roll(vals, s), index=lab.index)
        null_d.append(diff_share(lb))
        null_p.append(persistence(lb))
    null_d, null_p = np.array(null_d), np.array(null_p)
    return {"diff_share": obs_d, "diff_p": float(np.mean(null_d >= obs_d)),
            "persistence": obs_p, "persist_p": float(np.mean(null_p >= obs_p)), "n_null": len(null_p)}


# ─── 평가 ───
PERIODS = {"2012-2019": ("2012", "2019"), "2020-2024": ("2020", "2024"), "2025-2026": ("2025", "2026"),
           "ALL": ("2012", "2026")}


def perf(r):
    from engine import metrics
    return {p: metrics(r.loc[a:b]) for p, (a, b) in PERIODS.items()}


def sharpe_diff_p(a, b, block=6, reps=2000, seed=0):
    """월수익률 샤프 차이(a-b)의 블록 부트스트랩 단측 p값 (H0: 차이 ≤ 0)."""
    ma = (1 + a).groupby(a.index.to_period("M")).prod() - 1
    mb = (1 + b).groupby(b.index.to_period("M")).prod() - 1
    j = pd.concat([ma, mb], axis=1, join="inner").dropna().to_numpy()
    n = len(j)
    rng = np.random.default_rng(seed)
    sh = lambda x: x.mean(0) / x.std(0)
    obs = sh(j)[0] - sh(j)[1]
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, n // block + 1)])[:n]
        x = j[idx]
        dd = sh(x)
        diffs.append(dd[0] - dd[1])
    diffs = np.array(diffs)
    return float(obs * np.sqrt(12)), float(np.mean(diffs - diffs.mean() + 0 >= obs))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(running, 1.0)
    return adj


def run():
    d, cash, months = load_inputs()
    blocks = month_blocks(d, months)
    labels = RG.load()
    out, logs = {}, {}
    for mode in ("none", "d1", "d2", "d3", "d4"):
        out[mode], logs[mode] = walk_forward(mode, d, cash, months, blocks, labels)
        logs[mode].to_csv(RES / f"meta_log_{mode}.csv", index=False, encoding="utf-8-sig")
    out["KODEX200"], out["EW_all"] = baselines(d, cash, months, blocks)
    daily = pd.DataFrame(out)
    daily.to_parquet(RES / "meta_daily.parquet")

    rows = []
    for k, r in daily.items():
        for p, m in perf(r.dropna()).items():
            rows.append({"mode": k, "period": p, **{x: m.get(x) for x in ("CAGR", "Sharpe", "MDD", "Vol")}})
    table = pd.DataFrame(rows)
    table.to_csv(RES / "meta_perf.csv", index=False, encoding="utf-8-sig")

    # 판정: (c)=none 대비 샤프 차이, 구간별 + Holm
    tests = []
    for mode in ("d1", "d2", "d3", "d4"):
        a, b = daily[mode].dropna(), daily["none"].dropna()
        diff, p = sharpe_diff_p(a.loc["2012":"2024"], b.loc["2012":"2024"])
        s = lambda per: table[(table["mode"] == mode) & (table.period == per)].Sharpe.iloc[0] - \
            table[(table["mode"] == "none") & (table.period == per)].Sharpe.iloc[0]
        tests.append({"mode": mode, "sharpe_diff_12_24": diff, "p": p,
                      "diff_12_19": s("2012-2019"), "diff_20_24": s("2020-2024")})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p"].to_numpy())
    tests["pass"] = (tests["p_holm"] < 0.05) & (tests["diff_12_19"] > 0) & (tests["diff_20_24"] > 0)
    tests.to_csv(RES / "meta_tests.csv", index=False, encoding="utf-8-sig")

    # 국면 유의성 사전 검사
    mret = next_month_returns(d, months)
    sig = pd.DataFrame({m: regime_tests(labels[m], mret) for m in ("d1", "d2", "d3", "d4")}).T
    sig.to_csv(RES / "meta_regime_significance.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(sig.round(3).to_string())
    print(table.pivot_table(index="mode", columns="period", values="Sharpe").round(2).to_string())
    print(table.pivot_table(index="mode", columns="period", values="CAGR").round(3).to_string())
    print(tests.round(3).to_string())
    return daily, table, tests, sig


def run_windows(modes=None, tag="window"):
    """채점 기간 비교(2026-09-24 사전 등록): A=none(전체 과거) 대비 B(w12·w36·w60)·C(mix12).
    합격 = 2012~24 샤프 차이 Holm 보정 유의 + 2012~19·2020~24 둘 다 A 초과. 전부 불합격이면 A 유지."""
    d, cash, months = load_inputs()
    blocks = month_blocks(d, months)
    out, logs = {}, {}
    modes = modes or WINDOW_MODES
    for mode in ("none", *modes):
        out[mode], logs[mode] = walk_forward(mode, d, cash, months, blocks, None)
        logs[mode].to_csv(RES / f"meta_log_{mode}.csv", index=False, encoding="utf-8-sig")
    daily = pd.DataFrame(out)
    daily.to_parquet(RES / f"meta_{tag}_daily.parquet")
    rows = [{"mode": k, "period": p, **{x: m.get(x) for x in ("CAGR", "Sharpe", "MDD")}}
            for k, r in daily.items() for p, m in perf(r.dropna()).items()]
    table = pd.DataFrame(rows)
    tests = []
    for mode in modes:
        a, b = daily[mode].dropna(), daily["none"].dropna()
        diff, p = sharpe_diff_p(a.loc["2012":"2024"], b.loc["2012":"2024"])
        g = table.set_index(["mode", "period"]).Sharpe
        churn = logs[mode].picks.fillna("").str.split(",").apply(lambda x: set(filter(None, x)))
        tests.append({"mode": mode, "sharpe_diff_12_24": diff, "p": p,
                      "diff_12_19": g[mode, "2012-2019"] - g["none", "2012-2019"],
                      "diff_20_24": g[mode, "2020-2024"] - g["none", "2020-2024"],
                      "monthly_swaps": np.mean([len(b_ - a_) for a_, b_ in zip(churn[:-1], churn[1:])])})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p"].to_numpy())
    tests["pass"] = (tests["p_holm"] < 0.05) & (tests["diff_12_19"] > 0) & (tests["diff_20_24"] > 0)
    table.to_csv(RES / f"meta_{tag}_perf.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(RES / f"meta_{tag}_tests.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(table.pivot_table(index="mode", columns="period", values="Sharpe").round(2).to_string())
    print(table.pivot_table(index="mode", columns="period", values="CAGR").round(3).to_string())
    print(table.pivot_table(index="mode", columns="period", values="MDD").round(3).to_string())
    print(tests.round(3).to_string())
    for m in modes:
        print(m, logs[m].picks.iloc[-1])


if __name__ == "__main__":
    import sys
    if "windows" in sys.argv:
        run_windows()
    elif "ranks" in sys.argv:        # 순위 기준 비교(2026-09-24 사전 등록, Holm 3개)
        run_windows(RANK_MODES, "rank")
    else:
        run()
