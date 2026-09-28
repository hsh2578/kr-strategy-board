"""전 전략 실행 → results/metrics.csv, results/daily.parquet

    python run_all.py              # 전부 (이미 끝난 전략은 results/cache 에서 재사용)
    python run_all.py ON01 SW06    # 일부만 다시
    python run_all.py --jobs 4     # 전략 단위 4프로세스 병렬
"""
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import costs
import engine as E
import kr_data as K
import strategies as S

OUT = Path(__file__).parent / ("results" + (f"_{os.environ['BT_TAG']}" if os.environ.get("BT_TAG") else ""))
FAST = bool(os.environ.get("BT_TAG"))   # 민감도 실행은 비용2배·랜덤 검증 생략
CACHE = OUT / "cache"
OOS = pd.Timestamp("2020-01-01")
TAX = 0.154   # 국내상장 과세 ETF 매매차익 배당소득세(근사: 연말에 그해 과세 자산 기여 이익의 15.4% 차감)
STOCK_START = pd.Timestamp("2011-01-01")   # 지표 워밍업 1년


def stock_px():
    return {k: K.load(f"adj_{k}") for k in ("open", "high", "low", "close")}


def windows(daily, trades, slots=10):
    out = {}
    for tag, lo, hi in (("ALL", None, None), ("IS", None, OOS - pd.Timedelta(days=1)), ("OOS", OOS, None),
                        ("TO24", None, pd.Timestamp("2024-12-31"))):   # 2025~26 급등장 제외 구간
        dd = daily.loc[lo:hi]
        if len(dd) < 60:
            continue
        tr = None
        if trades is not None:
            t = trades["t_in"]
            tr = trades[(t >= (lo or t.min())) & (t <= (hi or t.max()) + pd.Timedelta(days=1))]
        out.update({f"{tag}_{k}": v for k, v in E.metrics(dd, tr, slots).items()})
    return out


def etf_tax(pf, daily, px):
    """과세 ETF 기여 이익에 연 1회 15.4% 차감(손익은 그해 안에서만 상계, 미실현 이익도 과세 → 보수적 근사)."""
    tax_cols = [c for c in px["close"].columns if c in K.ETFS and c not in K.DOMESTIC_EQUITY_ETFS]
    if not tax_cols:
        return daily
    val = pf.value()
    av = pf.asset_value(group_by=False)
    tax_cols = [c for c in tax_cols if c in av.columns]      # 한 번도 보유하지 않은 열 제외
    if not tax_cols:
        return daily
    held = av[tax_cols].div(val, axis=0).shift(1)
    r = px["close"][tax_cols].reindex(val.index).ffill().pct_change()
    contrib = (held * r).sum(axis=1).reindex(daily.index).fillna(0.0)
    out = daily.copy()
    for _, g in contrib.groupby(contrib.index.year):
        gain = g.sum()
        if gain > 0:
            out.loc[g.index[-1]] -= TAX * gain
    return out


def randomize(spec, seed=0):
    """랜덤 진입 대조군: 날짜별 진입 수는 그대로, 종목만 그날 유니버스에서 무작위."""
    rng = np.random.default_rng(seed)
    ent = spec.entry
    pool = S.d.U.reindex_like(ent).fillna(False).to_numpy()
    n = ent.sum(axis=1).to_numpy()
    out = np.zeros(ent.shape, dtype=bool)
    for i in np.nonzero(n)[0]:
        cand = np.nonzero(pool[i])[0]
        if len(cand):
            out[i, rng.choice(cand, size=min(n[i], len(cand)), replace=False)] = True
    kw = {k: v for k, v in spec.__dict__.items() if k != "entry"}
    return E.Spec(entry=pd.DataFrame(out, index=ent.index, columns=ent.columns), **kw)


def run_one(sid):
    info = S.REG[sid]
    res = info["fn"]()
    start = pd.Timestamp(info["since"]) if info["since"] else None
    row = {"id": sid, "cat": info["cat"], "name": info["name"], "src": info["src"],
           "assume": info["assume"], "etf": info["etf"]}

    if isinstance(res, dict):                                  # 리밸런싱형
        px = {k.lower(): v for k, v in res["px"].items()} if res.get("px") is not None else stock_px()
        idx = px["close"].index
        fee = costs.fee_rate(idx, etf=res["etf"])
        slip = (pd.DataFrame(costs.ETF_SLIP, index=idx, columns=px["close"].columns) if res["etf"]
                else costs.slippage(S.d.CAP.shift(1)))
        etf_cols = [c for c in px["close"].columns if c in K.ETFS]
        if not res["etf"] and etf_cols:   # 주식+ETF 혼합(할로윈·시장필터): ETF 열은 ETF 수수료·슬리피지(거래세 없음)
            fee = pd.DataFrame({c: fee for c in px["close"].columns})
            fee[etf_cols] = costs.COMMISSION
            slip = slip.reindex(columns=px["close"].columns)
            slip[etf_cols] = costs.ETF_SLIP
        runs = {}
        for tag, m in (("base", 1),) + ((("cost2x", 2),) if not FAST else ()):
            pf = E.rebalance(res["weights"], px, fee * m, slip * m)
            daily = etf_tax(pf, pf.returns(), px)
            first = daily.ne(0).idxmax()
            daily = daily.loc[max(first, start) if start is not None else first:]
            runs[tag] = (pf, daily)
        pf, daily = runs["base"]
        row.update(windows(daily, None))
        row["basis"] = "월간"
        row["Turnover_rb"] = E.rebalance_turnover(pf)
        if not FAST:
            row["cost2x_OOS_Sharpe"] = E.metrics(runs["cost2x"][1].loc[OOS:]).get("Sharpe", np.nan)
            row["cost2x_ALL_CAGR"] = E.metrics(runs["cost2x"][1])["CAGR"]
        return row, daily

    spec = res                                                 # 신호형
    if spec.etf:
        px = S.etf_px(*spec.meta["px"])
        idx = px["close"].index
        fee = costs.fee_rate(idx, etf=True)
        slip = pd.DataFrame(costs.ETF_SLIP, index=idx, columns=px["close"].columns)
        tv = pd.DataFrame(1.0, index=idx, columns=px["close"].columns)
    else:
        px = stock_px()
        idx = px["close"].index
        fee = costs.fee_rate(idx)
        slip = costs.slippage(S.d.CAP.shift(1))
        tv = S.d.TV
        start = max(start, STOCK_START) if start is not None else STOCK_START

    def go(sp, m=1):
        pf = E.simulate(sp, px, fee * m, slip * m)
        if pf is None:
            return pd.Series(0.0, index=idx), None
        kept = E.allocate(pf, sp, tv)
        daily = E.portfolio_returns(pf, kept, sp.slots)
        return daily, kept

    daily, kept = go(spec)
    lo = start if start is not None else daily.ne(0).idxmax()
    daily = daily.loc[lo:]
    kept = kept[kept["t_in"] >= lo] if kept is not None else pd.DataFrame(columns=["t_in", "t_out", "Return"])
    row.update(windows(daily, kept, spec.slots))
    row["basis"] = "거래"
    if FAST:
        return row, daily
    d2, _ = go(spec, 2)
    row["cost2x_OOS_Sharpe"] = E.metrics(d2.loc[OOS:]).get("Sharpe", np.nan)
    row["cost2x_ALL_CAGR"] = E.metrics(d2.loc[lo:])["CAGR"]
    if not spec.etf and spec.entry_at != "intraday":  # 장중 지정가는 종목별 체결가가 달라 랜덤 비교 불가
        dr, _ = go(randomize(spec))
        row["random_OOS_Sharpe"] = E.metrics(dr.loc[OOS:]).get("Sharpe", np.nan)
        row["random_ALL_CAGR"] = E.metrics(dr.loc[lo:])["CAGR"]
    return row, daily


def work(sid):
    t = time.time()
    try:
        row, daily = run_one(sid)
        pd.to_pickle((row, daily), CACHE / f"{sid}.pkl")  # 이 스크립트가 만든 로컬 캐시
        return (f"{sid} {time.time() - t:5.0f}s  OOS CAGR {row.get('OOS_CAGR', np.nan):.1%}  "
                f"Sharpe {row.get('OOS_Sharpe', np.nan):.2f}  MDD {row.get('OOS_MDD', np.nan):.1%}")
    except Exception:
        return f"{sid} 실패\n{traceback.format_exc()}"


def main(ids, jobs=1):
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = ids or [s for s in S.REG if not (CACHE / f"{s}.pkl").exists()]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            for msg in ex.map(work, todo):
                print(msg, flush=True)
    else:
        for sid in todo:
            print(work(sid), flush=True)
    rows, dailies = [], {}
    for sid in S.REG:
        f = CACHE / f"{sid}.pkl"
        if f.exists():
            row, daily = pd.read_pickle(f)
            rows.append(row)
            dailies[sid] = daily
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dailies).to_parquet(OUT / "daily.parquet")
    print(f"저장: {OUT / 'metrics.csv'} ({len(rows)}개)")


if __name__ == "__main__":
    args = sys.argv[1:]
    jobs = 1
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    main(args, jobs)
