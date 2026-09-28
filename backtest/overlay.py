"""조합 위에 얹는 노출·가중 조절 3종 검증 (2026-09-25 사전 등록, 결과 보기 전 고정).

기준 조합(BASE): 1·2등급 6개(LT32 AA17 AA09 AA10 AA02 AA03) 동일비중(그날 운용 중인 전략 평균).
  M1 변동성 타기팅: 노출 = min(1, 목표 / EWMA(반감 20일) 연율 변동성), 나머지 단기채. 목표 6·8·10%.
  M2 전략별 모멘텀 필터: 월말 각 전략 최근 초과수익 > 0 이면 켬, 켜진 전략 60일 역변동성 가중, 없으면 단기채.
     판단 = 6개월 초과 / 12개월 초과 / 12개월 KODEX200 잔차.
  M3 점프모형 노출: 약세(d3) 달엔 노출 1 - k, k = 0.25·0.5·1.0.
공통: 신호 1일 지연, 노출 변경 |Δ|×0.1%, 전략 비중 변경 |Δ|×0.2%. 선택 = 2011~2019 초과 샤프 최고,
OOS 2020~2024 · 2025~26 은 한 번만 본다. 9개 변형 + 기준으로 CSCV PBO(S=16).

    python overlay.py      # results/overlay_*.csv
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import engine as E
import regimes as RG

RES = Path(__file__).parent / "results"
BASE_IDS = ["LT32", "AA17", "AA09", "AA10", "AA02", "AA03"]
C_EXPO, C_W = 0.001, 0.002


def excess_sharpe(r, rf):
    e = (r - rf.reindex(r.index)).dropna()
    return e.mean() / e.std() * np.sqrt(252) if e.std() > 0 else np.nan


def cagr(r):
    r = r.dropna()
    return (1 + r).prod() ** (252 / len(r)) - 1


def mdd(r):
    eq = (1 + r.dropna()).cumprod()
    return (eq / eq.cummax() - 1).min()


def expo_overlay(base, rf, expo):
    """노출 e_t(전일 결정) × 기준 + (1-e) × 단기채 − 노출 변경 비용."""
    e = expo.reindex(base.index).ffill().shift(1).fillna(1.0).clip(0, 1)
    return e * base + (1 - e) * rf.reindex(base.index) - e.diff().abs().fillna(0) * C_EXPO


def m1(base, rf, target):
    vol = base.ewm(halflife=20).std() * np.sqrt(252)
    return expo_overlay(base, rf, (target / vol).clip(upper=1.0))


def m2(R, rf, kind):
    ex = R.sub(rf.reindex(R.index), axis=0)
    me = R.index.to_series().groupby(R.index.to_period("M")).last()
    if kind == "resid12":
        mk = (E.rf_daily(R.index) * 0 + pd.read_parquet(RES / "daily.parquet")["H01"].reindex(R.index)) - rf.reindex(R.index)
        sig = pd.DataFrame(index=me.values, columns=R.columns, dtype=float)
        for t in me.values:
            w = ex.loc[:t].tail(252)
            for c in R.columns:
                z = pd.concat([w[c], mk.loc[w.index]], axis=1).dropna()
                if len(z) < 200:
                    continue
                b = np.polyfit(z.iloc[:, 1], z.iloc[:, 0], 1)
                sig.at[t, c] = (z.iloc[:, 0] - b[0] * z.iloc[:, 1]).sum()
    else:
        n = 126 if kind == "ex6" else 252
        sig = ex.rolling(n, min_periods=int(n * 0.9)).sum().reindex(me.values)
    on = sig > 0
    iv = 1 / R.rolling(60, min_periods=40).std().reindex(me.values)
    w = (iv * on).div((iv * on).sum(axis=1), axis=0).fillna(0.0)
    wd = w.reindex(R.index).ffill().shift(1).fillna(0.0)
    cash = 1 - wd.sum(axis=1)
    r = (wd * R.fillna(0)).sum(axis=1) + cash * rf.reindex(R.index)
    return r - wd.diff().abs().sum(axis=1).fillna(0) * C_W


def m3(base, rf, k):
    lab = RG.load()["d3"]
    bear = pd.Series((lab == "약세").astype(float).to_numpy(), index=lab.index.to_timestamp(how="end").normalize())
    expo = 1 - k * bear
    # 월말 라벨 → 다음 거래일부터(1일 지연은 expo_overlay 가 추가)
    return expo_overlay(base, rf, expo.reindex(base.index.union(expo.index)).ffill().reindex(base.index))


def pbo(P: pd.DataFrame, S=16):
    """CSCV: 월별 블록 S개, IS 절반 조합마다 IS 최고 변형의 OOS 상대순위 < 중앙이면 과적합."""
    mo = (1 + P).groupby(P.index.to_period("M")).prod() - 1
    blocks = np.array_split(np.arange(len(mo)), S)
    logits = []
    for isb in combinations(range(S), S // 2):
        ii = np.concatenate([blocks[b] for b in isb])
        oo = np.concatenate([blocks[b] for b in range(S) if b not in isb])
        sh = lambda idx: mo.iloc[idx].mean() / mo.iloc[idx].std()
        best = sh(ii).idxmax()
        rank = sh(oo).rank(pct=True)[best]
        logits.append(np.log(rank / (1 - rank + 1e-9) + 1e-9))
    return float(np.mean(np.array(logits) <= 0))


def main():
    d = pd.read_parquet(RES / "daily.parquet")
    R = d[BASE_IDS].loc["2011":]
    rf = E.rf_daily(R.index)
    base = R.mean(axis=1, skipna=True)
    V = {"BASE": base}
    for t in (0.06, 0.08, 0.10):
        V[f"M1_vol{int(t * 100)}"] = m1(base, rf, t)
    for kind in ("ex6", "ex12", "resid12"):
        V[f"M2_{kind}"] = m2(R, rf, kind)
    for k in (0.25, 0.5, 1.0):
        V[f"M3_k{k}"] = m3(base, rf, k)
    V["BM_KODEX200"] = d["H01"].loc[R.index]
    V["BM02_45_55"] = d["BM02"].loc[R.index]
    P = pd.DataFrame(V)
    P.to_parquet(RES / "overlay_daily.parquet")
    rows = []
    for k, r in P.items():
        row = {"variant": k}
        for per, a, b in (("IS11_19", "2011", "2019"), ("OOS20_24", "2020", "2024"), ("OOS25_26", "2025", "2026")):
            x = r.loc[a:b]
            row.update({f"{per}_CAGR": cagr(x), f"{per}_Sharpe": excess_sharpe(x, rf), f"{per}_MDD": mdd(x)})
        rows.append(row)
    T = pd.DataFrame(rows).set_index("variant")
    picks = {m: T.loc[[i for i in T.index if i.startswith(m)], "IS11_19_Sharpe"].idxmax() for m in ("M1", "M2", "M3")}
    variants = [c for c in P.columns if c.startswith(("M1", "M2", "M3", "BASE"))]
    T["picked_by_IS"] = T.index.isin(picks.values())
    T.to_csv(RES / "overlay_perf.csv", encoding="utf-8-sig")
    p = pbo(P[variants].loc["2011":"2024"].dropna())
    pd.set_option("display.width", 220)
    print(T.round(3).to_string())
    print("IS 로 고른 변형:", picks)
    print(f"PBO(2011~2024, 변형 {len(variants)}개, S=16): {p:.2f}")
    return T, p


if __name__ == "__main__":
    main()
