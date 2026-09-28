"""엔진 최소 검증: 손으로 계산한 값과 vectorbt 결과가 같은지."""
import numpy as np
import pandas as pd
import pytest

import engine as E

IDX = pd.bdate_range("2024-01-01", periods=6)


def px(opens, closes, highs=None, lows=None, cols=("A",)):
    o = pd.DataFrame({c: opens for c in cols}, index=IDX, dtype=float)
    c = pd.DataFrame({c: closes for c in cols}, index=IDX, dtype=float)
    h = pd.DataFrame({c: highs or np.maximum(opens, closes) for c in cols}, index=IDX, dtype=float)
    lo = pd.DataFrame({c: lows or np.minimum(opens, closes) for c in cols}, index=IDX, dtype=float)
    return {"open": o, "high": h, "low": lo, "close": c}


def sig(days, cols=("A",)):
    s = pd.DataFrame(False, index=IDX, columns=list(cols))
    for d in days:
        s.iloc[d] = True
    return s


FEE = 0.001
fees = pd.Series(FEE, index=IDX)
noslip = pd.DataFrame(0.0, index=IDX, columns=["A", "B"])


def run(spec, p):
    pf = E.simulate(spec, p, fees, noslip)
    return pf.trades.records_readable


def test_next_open_entry_hold_one_day_cost():
    p = px([100, 100, 100, 100, 100, 100], [100, 100, 110, 100, 100, 100])
    tr = run(E.Spec(entry=sig([1]), hold=1), p)       # 1일 신호 → 2일 시가 100 매수, 2일 종가 110 청산
    assert len(tr) == 1
    expect = 110 * (1 - FEE) / (100 * (1 + FEE)) - 1
    assert tr["PnL"].iloc[0] == pytest.approx(expect, rel=1e-9)  # 현금 1.0 기준 손익
    assert tr["Entry Timestamp"].iloc[0] == IDX[2] + E.OPEN_T
    assert tr["Exit Timestamp"].iloc[0] == IDX[2] + E.CLOSE_T


def test_overnight_close_to_next_open():
    p = px([100, 100, 105, 100, 100, 100], [100, 100, 100, 100, 100, 100])
    tr = run(E.Spec(entry=sig([1]), entry_at="close", hold=1, exit_hold_at="open"), p)
    assert tr["Entry Timestamp"].iloc[0] == IDX[1] + E.CLOSE_T
    assert tr["Exit Timestamp"].iloc[0] == IDX[2] + E.OPEN_T
    assert tr["Avg Entry Price"].iloc[0] == 100 and tr["Avg Exit Price"].iloc[0] == 105


def test_limit_up_open_blocks_entry():
    p = px([100, 100, 130, 130, 130, 130], [100, 100, 130, 130, 130, 130])
    tr = run(E.Spec(entry=sig([1]), hold=1), p)       # 다음날 시가 +30% → 매수 불가
    assert tr.empty


def test_stop_loss_uses_intraday_low():
    p = px([100] * 6, [100] * 6, lows=[100, 100, 90, 100, 100, 100])
    tr = run(E.Spec(entry=sig([1]), sl=0.05, hold=3), p)
    assert tr["Avg Exit Price"].iloc[0] == pytest.approx(95)


def test_slots_limit_and_portfolio_return():
    cols = ("A", "B")
    p = px([100] * 6, [100, 100, 110, 110, 110, 110], cols=cols)
    spec = E.Spec(entry=sig([1], cols), hold=1, slots=1)
    pf = E.simulate(spec, p, fees, noslip)
    tv = pd.DataFrame({"A": 1.0, "B": 2.0}, index=IDX)   # B 우선
    kept = E.allocate(pf, spec, tv)
    assert list(kept["col"]) == ["B"]
    daily = E.portfolio_returns(pf, kept, slots=1)
    assert daily.loc[IDX[2]] == pytest.approx(110 * (1 - FEE) / (100 * (1 + FEE)) - 1)


def test_intraday_priority_uses_previous_day():
    """장중 진입의 슬롯 우선순위에 당일 거래대금(장 마감 후에야 앎)을 쓰면 안 된다."""
    cols = ("A", "B")
    p = px([100] * 6, [100, 100, 110, 110, 110, 110], cols=cols)
    spec = E.Spec(entry=sig([2], cols), entry_at="intraday",
                  entry_price=pd.DataFrame(100.0, index=IDX, columns=list(cols)), hold=1, slots=1)
    pf = E.simulate(spec, p, fees, noslip)
    tv = pd.DataFrame({"A": 1.0, "B": 1.0}, index=IDX)
    tv.loc[IDX[1], "A"] = 5.0   # 전일: A 우선
    tv.loc[IDX[2], "B"] = 9.0   # 당일: B 가 크지만 아직 모르는 값
    assert list(E.allocate(pf, spec, tv)["col"]) == ["A"]


def test_close_entry_blocked_at_limit_up_close():
    p = px([100] * 6, [100, 100, 130, 130, 130, 130])
    tr = run(E.Spec(entry=sig([2]), entry_at="close", hold=1), p)
    assert tr.empty


# ─── 2026-09-24 감사 후 추가: 당일청산·거래정지·손절 슬리피지·리밸런싱 상장폐지 ───
def test_intraday_entry_exits_same_day_close():
    p = px([100, 100, 100, 100, 100, 100], [100, 110, 100, 100, 100, 100], highs=[100, 112, 100, 100, 100, 100])
    ep = pd.DataFrame({"A": [np.nan, 102, np.nan, np.nan, np.nan, np.nan]}, index=IDX)
    tr = run(E.Spec(entry=sig([1]), entry_at="intraday", entry_price=ep, hold=1), p)
    assert tr["Avg Entry Price"].iloc[0] == 102
    assert tr["Exit Timestamp"].iloc[0] == IDX[1] + E.CLOSE_T     # 같은 날 종가
    assert tr["Avg Exit Price"].iloc[0] == 110


def test_time_stop_waits_while_halted():
    p = px([100, 100, np.nan, 10, 10, 10], [100, 100, 100, 10, 10, 10])   # 2일 정지(종가칸엔 직전가), 3일 정리매매
    p["high"].iloc[2] = p["low"].iloc[2] = np.nan
    tr = run(E.Spec(entry=sig([0]), hold=2), p)        # 1일 시가 매수, 원래 2일 종가 청산
    assert tr["Exit Timestamp"].iloc[0] == IDX[3] + E.OPEN_T         # 정지 풀린 첫 봉
    assert tr["Avg Exit Price"].iloc[0] == 10


def test_close_entry_blocked_while_halted():
    p = px([100, np.nan, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100])
    p["high"].iloc[1] = p["low"].iloc[1] = np.nan
    tr = run(E.Spec(entry=sig([1]), entry_at="close", hold=1), p)
    assert tr.empty


def test_stop_exit_pays_slippage():
    p = px([100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100], lows=[100, 100, 90, 100, 100, 100])
    slip = pd.DataFrame(0.01, index=IDX, columns=["A", "B"])
    pf = E.simulate(E.Spec(entry=sig([0]), sl=0.05), p, fees, slip)
    tr = pf.trades.records_readable
    assert tr["Avg Exit Price"].iloc[0] == pytest.approx(100 * 1.01 * 0.95 * 0.99)   # 체결가 101 기준 손절가에서 슬리피지 1%


def _rb_px(a_open, a_close, b=100.0):
    o = pd.DataFrame({"A": a_open, "B": [b] * 6}, index=IDX, dtype=float)
    c = pd.DataFrame({"A": a_close, "B": [b] * 6}, index=IDX, dtype=float)
    return {"open": o, "high": o, "low": o, "close": c}


def test_rebalance_delisted_turns_to_cash_at_last_close():
    p = _rb_px([100, 100, 50, np.nan, np.nan, np.nan], [100, 100, 40, np.nan, np.nan, np.nan])  # 2일이 마지막 거래(정리매매)
    w = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[IDX[0]])
    pf = E.rebalance(w, p, pd.Series(0.0, index=IDX), pd.DataFrame(0.0, index=IDX, columns=["A", "B"]))
    assert pf.value().iloc[-1] == pytest.approx(0.4)          # 100 매수 → 40 에 청산
    assert pf.cash().iloc[-1] == pytest.approx(0.4)           # 현금으로 돌아옴


def test_rebalance_sell_retries_after_halt():
    p = _rb_px([100, 100, np.nan, 80, 80, 80], [100, 100, 100, 80, 80, 80])
    w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[IDX[0], IDX[1]])   # 1일 신호 → 2일 매도 예정, 2일 정지
    pf = E.rebalance(w, p, pd.Series(0.0, index=IDX), pd.DataFrame(0.0, index=IDX, columns=["A", "B"]))
    sells = pf.orders.records_readable
    sells = sells[(sells["Side"] == "Sell")]
    assert sells["Timestamp"].iloc[0] == IDX[3] and sells["Price"].iloc[0] == 80


def test_intraday_overnight_exits_next_open():
    p = px([100, 100, 107, 100, 100, 100], [100, 104, 100, 100, 100, 100], highs=[100, 106, 107, 100, 100, 100])
    ep = pd.DataFrame({"A": [np.nan, 102, np.nan, np.nan, np.nan, np.nan]}, index=IDX)
    tr = run(E.Spec(entry=sig([1]), entry_at="intraday", entry_price=ep, hold=1, exit_hold_at="open"), p)
    assert tr["Exit Timestamp"].iloc[0] == IDX[2] + E.OPEN_T and tr["Avg Exit Price"].iloc[0] == 107


def test_next_open_hold_one_exit_open_is_next_day():
    p = px([100, 100, 103, 106, 100, 100], [100, 101, 104, 100, 100, 100])
    tr = run(E.Spec(entry=sig([0]), hold=1, exit_hold_at="open"), p)
    assert tr["Exit Timestamp"].iloc[0] == IDX[2] + E.OPEN_T and tr["Avg Exit Price"].iloc[0] == 103


def test_intraday_tp_ignores_high_before_entry():
    # 시가 110 에서 밀려 100 에 지정가 매수, 종가 100 → 당일 고가 110 은 매수 전이므로 익절(101) 아님
    p = px([100, 110, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100], highs=[100, 110, 100, 100, 100, 100],
           lows=[100, 99, 100, 100, 100, 100])
    ep = pd.DataFrame({"A": [np.nan, 100, np.nan, np.nan, np.nan, np.nan]}, index=IDX)
    tr = run(E.Spec(entry=sig([1]), entry_at="intraday", entry_price=ep, tp=0.01, hold=3), p)
    assert tr["Exit Timestamp"].iloc[0] != IDX[1] + E.CLOSE_T


def test_intraday_exit_at_open_not_overwritten_by_next_entry():
    # 1일 돌파 매수(102) → 2일 시가(100) 청산. 2일에도 돌파 신호(105)가 있어도 청산가는 시가 100
    p = px([100, 100, 100, 100, 100, 100], [100, 103, 104, 100, 100, 100], highs=[100, 103, 106, 100, 100, 100])
    ep = pd.DataFrame({"A": [np.nan, 102, 105, np.nan, np.nan, np.nan]}, index=IDX)
    tr = run(E.Spec(entry=sig([1, 2]), entry_at="intraday", entry_price=ep, hold=1, exit_hold_at="open"), p)
    assert tr["Avg Exit Price"].iloc[0] == 100
    assert tr["Avg Entry Price"].iloc[1] == 105        # 같은 날 다시 장중 매수 가능


# ─── 2026-09-25 최종 검증 후: 상장폐지 판정(B1), 혼합 비용(B2), 초과 샤프 ───
def _oc(opens, closes):
    return (pd.DataFrame({"A": opens}, index=IDX, dtype=float), pd.DataFrame({"A": closes}, index=IDX, dtype=float))


def test_halted_until_end_is_not_delisted():
    o, c = _oc([100, 100, np.nan, np.nan, np.nan, np.nan], [100, 100, 100, 100, 100, 100])   # 끝까지 정지(아직 상장)
    assert E.delist_exits(o, c) == {}


def test_delist_after_liquidation_trading_uses_last_close():
    o, c = _oc([100, np.nan, 10, 8, np.nan, np.nan], [100, 100, 10, 7, np.nan, np.nan])
    assert E.delist_exits(o, c) == {"A": (IDX[3], 7.0)}


def test_delist_without_liquidation_short_halt_keeps_price(monkeypatch):
    o, c = _oc([100, 100, np.nan, np.nan, np.nan, np.nan], [100, 100, 100, 100, np.nan, np.nan])   # 합병형 짧은 정지
    assert E.delist_exits(o, c) == {"A": (IDX[3], 100.0)}
    monkeypatch.setattr(E, "LONG_HALT", 2)                                                         # 장기 정지로 간주
    assert E.delist_exits(o, c)["A"][1] == pytest.approx(100 * (1 + E.DELIST_HAIRCUT))


def test_rebalance_accepts_per_column_fees():
    p = _rb_px([100.0] * 6, [100.0] * 6)
    w = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[IDX[0]])
    fees = pd.DataFrame({"A": 0.01, "B": 0.0}, index=IDX)
    pf = E.rebalance(w, p, fees, pd.DataFrame(0.0, index=IDX, columns=["A", "B"]))
    od = pf.orders.records_readable.set_index("Column")
    assert od.loc["A", "Fees"] > 0 and od.loc["B", "Fees"] == 0


def test_metrics_sharpe_is_excess_over_rf():
    idx = pd.bdate_range("2015-01-02", periods=500)
    rf = E.rf_daily(idx)
    m = E.metrics(rf.copy())                       # 단기채 자체를 들고 있으면 초과수익 0
    assert abs(m["Sharpe"]) < 1e-6 or np.isnan(m["Sharpe"])
