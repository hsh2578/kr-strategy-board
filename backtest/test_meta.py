"""메타 시스템 최소 검증: 미래 누설 없음, 슬롯 20%·단기채 배분, 국면 표본이 과거만 사용."""
import numpy as np
import pandas as pd

import meta as M


def _toy(n_months=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n_months * 21)
    idx = idx[idx.to_period("M") <= idx.to_period("M")[0] + n_months - 1]
    d = pd.DataFrame({"GOOD": 0.001 + rng.normal(0, 0.005, len(idx)),
                      "OK": 0.0006 + rng.normal(0, 0.005, len(idx)),
                      "BAD": -0.001 + rng.normal(0, 0.01, len(idx))}, index=idx)
    months = idx.to_period("M")
    cash = pd.Series(0.0001, index=idx)
    return d, cash, months


def test_no_future_leak():
    d, cash, months = _toy()
    blocks = M.month_blocks(d, months)
    ms = sorted(blocks)
    t = ms[20]
    base, _ = M.select(t, None, blocks, ms, list(d.columns))
    d2 = d.copy()
    d2.loc[months > t, "BAD"] = 0.05                  # 미래에 BAD 가 폭등해도
    d2.loc[months > t, "GOOD"] = -0.05
    b2 = M.month_blocks(d2, months)
    again, _ = M.select(t, None, b2, ms, list(d.columns))
    assert [p[0] for p in base] == [p[0] for p in again]


def test_slots_and_cash():
    d, cash, months = _toy()
    blocks = M.month_blocks(d, months)
    daily, log = M.walk_forward("none", d, cash, months, blocks, None)
    row = log.iloc[-1]
    n = len(row.picks.split(",")) if row.picks else 0
    assert "BAD" not in row.picks
    assert abs(row.cash_w - (1 - 0.2 * n)) < 1e-9
    # 첫 12개월은 이력 부족 → 전량 단기채
    assert log.iloc[0].picks == "" and log.iloc[0].cash_w == 1.0
    hold = pd.Period(log.iloc[0].hold)
    assert np.allclose(daily[daily.index.to_period("M") == hold], 0.0001)


def test_regime_sample_uses_past_only():
    d, cash, months = _toy()
    blocks = M.month_blocks(d, months)
    ms = sorted(blocks)
    labels = pd.Series(["A" if i % 2 else "B" for i in range(len(ms))], index=pd.PeriodIndex(ms))
    t = ms[30]
    _, n = M.select(t, labels, blocks, ms, list(d.columns))
    # t 의 국면과 같은 국면이었던 달 m(≤t-1)의 다음 달 수
    expect = sum(1 for m in ms if m + 1 <= t and labels[m] == labels[t])
    assert n == expect
