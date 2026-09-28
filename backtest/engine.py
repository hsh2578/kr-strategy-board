"""vectorbt 실행 래퍼 + 한국시장 규칙.

체결·손절·익절·수수료·슬리피지·성과지표 계산은 vectorbt 가 한다. 이 파일은
1) 하루를 시가·종가 두 시점으로 나눈 가격열을 만들어 오버나잇·종가 진입도 vectorbt 안에서 처리하고
2) 한국 규칙(상한가 출발 매수 차단, 거래정지 무체결, 상장폐지 강제청산)을 신호에 반영하고
3) 종목별 vectorbt 체결 결과를 '최대 N종목 동일비중' 포트폴리오로 묶는다(슬롯 배분).
   ponytail: 슬롯은 매일 1/N 동일비중으로 재조정된다고 가정(일간 수익률 평균). 정밀 비중 추적이 필요하면 from_order_func 로 교체.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit

vbt.settings.returns["year_freq"] = "252 days"
OPEN_T, CLOSE_T = pd.Timedelta(hours=9), pd.Timedelta(hours=15, minutes=30)
MID_T = pd.Timedelta(hours=12)   # 장중 진입 전략 전용 봉(시가 청산과 장중 진입가가 섞이지 않게)
LIMIT_UP_OPEN = 0.29
# 상장폐지 청산(2026-09-25): 정리매매 197건의 '정지 직전가 대비 최종가' 중앙값 -94.2%(사분위 -98% ~ -80%).
# 정리매매 가격 없이 폐지된 119건 중 116건은 정지 11~20거래일(합병·공개매수) → 60거래일 이상만 부실로 본다.
DELIST_HAIRCUT, LONG_HALT = -0.942, 60
LIMIT_CHANGE = pd.Timestamp("2015-06-15")   # 가격제한폭 ±15% → ±30%


def delist_exits(o: pd.DataFrame, c: pd.DataFrame) -> dict:
    """상장폐지 종목 → (청산일, 청산가). 종가가 패널 끝 전에 끝난 종목만 폐지로 본다
    (끝까지 정지 중인 종목은 아직 상장 상태라 청산하지 않는다 — 정지 직전 청산은 미래참조).
    마지막 날 거래(정리매매)가 있으면 그 종가, 없으면 정지 60거래일 이상은 직전가×(1-94.2%), 짧으면(합병 등) 직전가."""
    out, end = {}, c.index[-1]
    for col in c.columns:
        lc = c[col].last_valid_index()
        if lc is None or lc >= end:
            continue
        lo = o[col].last_valid_index()
        px = float(c.at[lc, col])
        if lo is None or lo < lc:
            n = c.index.get_loc(lc) - (c.index.get_loc(lo) if lo is not None else 0)
            if n >= LONG_HALT:
                px *= 1 + DELIST_HAIRCUT
        out[col] = (lc, px)
    return out


_RF = None


def rf_daily(index: pd.DatetimeIndex) -> pd.Series:
    """무위험수익률 대용: KODEX 단기채권(153130) 일간수익률, 상장(2012-02) 전은 KODEX 국고채3년(114260)."""
    global _RF
    if _RF is None:
        import kr_data as K
        e = K.load("etf")
        r = e["153130"]["Close"].dropna().pct_change()
        _RF = r.combine_first(e["114260"]["Close"].dropna().pct_change()).fillna(0.0)
    return _RF.reindex(index).fillna(0.0)


def near_limit(index: pd.DatetimeIndex) -> pd.Series:
    """상한가로 간주하는 상승률(제한폭 - 1%p): 2015-06-15 이전 14%, 이후 29%."""
    return pd.Series(np.where(index < LIMIT_CHANGE, 0.14, LIMIT_UP_OPEN), index=index)


@dataclass
class Spec:
    """전략 신호 정의. 신호는 모두 '그날 장 마감 후 알 수 있는 정보'로 만든다(일봉 인덱스)."""
    entry: pd.DataFrame                  # bool
    entry_at: str = "next_open"          # next_open | close(당일 종가) | intraday(당일 entry_price)
    entry_price: pd.DataFrame = None     # intraday 진입가(당일 장중 돌파가 등)
    exit: pd.DataFrame = None            # bool, 신호일 기준
    exit_at: str = "next_open"           # next_open | close(신호 당일 종가)
    hold: int = None                     # 최대 보유 거래일
    exit_hold_at: str = "close"          # 보유기간 만료일의 close | open
    sl: object = None                    # 손절 비율(스칼라 또는 신호일 기준 DataFrame)
    tp: object = None
    trail: bool = False
    priority: pd.DataFrame = None        # 슬롯 경쟁 시 우선순위(클수록 먼저). 기본: 신호일 거래대금
    slots: int = 10
    etf: bool = False
    meta: dict = field(default_factory=dict)


def _grid(daily_index: pd.DatetimeIndex, mid: bool = False) -> pd.DatetimeIndex:
    g = (daily_index + OPEN_T).append(daily_index + CLOSE_T)
    return (g.append(daily_index + MID_T) if mid else g).sort_values()


def _on(daily: pd.DataFrame, when: str, idx2, fill=np.nan) -> pd.DataFrame:
    """일봉 프레임을 격자의 open·mid·close 시점에 놓는다."""
    d = daily.set_axis(daily.index + {"open": OPEN_T, "mid": MID_T, "close": CLOSE_T}[when])
    return d.reindex(idx2, fill_value=fill) if fill is not np.nan else d.reindex(idx2)


@njit
def _signal_func(c, entries, exits, hold, entry_i, halted, pending):
    """진입/청산 신호 + 최대 보유 봉수(time stop). c.position_now 는 이 봉 주문 전 포지션.
    거래정지 봉에서는 아무 주문도 내지 않고, 그때 걸린 청산은 정지가 풀린 첫 봉으로 미룬다."""
    if c.position_now > 0:
        if entry_i[c.col] < 0:
            entry_i[c.col] = c.i - 1
        want = pending[c.col] or exits[c.i, c.col] or (hold > 0 and c.i - entry_i[c.col] >= hold)
        if halted[c.i, c.col]:
            pending[c.col] = want
            return False, False, False, False
        pending[c.col] = False
        return False, want, False, False
    entry_i[c.col] = -1
    pending[c.col] = False
    if halted[c.i, c.col]:
        return False, False, False, False
    return entries[c.i, c.col], False, False, False


def simulate(spec: Spec, px: dict, fees: pd.Series, slip: pd.DataFrame):
    """종목별 독립 체결(종목마다 자기 자본 100% 투입). vectorbt Portfolio(2시점 격자) 반환."""
    cols = spec.entry.columns[spec.entry.any()]
    if len(cols) == 0:
        return None
    ent = spec.entry[cols].fillna(False).astype(bool)
    o, h, l, c = (px[k].reindex(columns=cols) for k in ("open", "high", "low", "close"))
    idx = c.index
    intra = spec.entry_at == "intraday"
    idx2 = _grid(idx, mid=intra)
    if intra:   # 시가봉(시가) → 장중봉(진입일은 진입가, 아니면 시가) → 종가봉
        epd = spec.entry_price.reindex(columns=cols)
        ent = ent & ~(epd.ge(c.shift(1).mul(1 + near_limit(idx), axis=0)))  # 상한가 가격 매수 불가
        m = epd.where(ent).combine_first(o)
        on3 = lambda a, b, z: _on(a, "open", idx2).combine_first(_on(b, "mid", idx2)).combine_first(_on(z, "close", idx2))
        # 진입일 고가는 매수 뒤인지 알 수 없다 → 종가봉 고가를 max(진입가, 종가)로 제한(보수적)
        hcap = h.mask(ent, np.minimum(h, np.maximum(epd, c)))
        O, H, L, C = on3(o, m, m), on3(o, m, hcap), on3(o, m, l), on3(o, m, c)
    else:
        O = _on(o, "open", idx2).combine_first(_on(o, "close", idx2))       # 두 봉 모두 시가
        H = _on(o, "open", idx2).combine_first(_on(h, "close", idx2))
        L = _on(o, "open", idx2).combine_first(_on(l, "close", idx2))
        C = _on(o, "open", idx2).combine_first(_on(c, "close", idx2))       # 시가봉 '종가'=시가
    halt = o.isna() & c.notna()                                           # 거래정지(종가칸엔 직전가가 남아 있음)
    HALT = _on(halt, "open", idx2, False) | _on(halt, "close", idx2, False)
    if intra:
        HALT |= _on(halt, "mid", idx2, False)

    # 진입 봉과 진입가
    if spec.entry_at == "next_open":
        lim = near_limit(idx).shift(-1).bfill()
        ent = ent & ~(o.shift(-1).ge(c.mul(1 + lim, axis=0)))            # 상한가 출발 매수 불가
        E = _on(ent.shift(1, fill_value=False), "open", idx2, False)
        price = C.copy()
    elif spec.entry_at == "close":
        ent = ent & ~(c.ge(c.shift(1).mul(1 + near_limit(idx), axis=0)))  # 상한가 마감 종목은 종가 매수 불가
        E = _on(ent, "close", idx2, False)
        price = C.copy()
    else:   # intraday: 장중봉에서 진입가로 체결(장중봉 C = 진입가)
        E = _on(ent, "mid", idx2, False)
        price = C.copy()
    E &= price.notna().to_numpy()                                         # 거래정지면 체결 없음

    # 청산 봉 (가격은 해당 봉의 C: 시가봉이면 시가, 종가봉이면 종가)
    X = pd.DataFrame(False, index=idx2, columns=cols)
    if spec.exit is not None:
        ex = spec.exit.reindex(columns=cols).fillna(False).astype(bool)
        X |= (_on(ex.shift(1, fill_value=False), "open", idx2, False) if spec.exit_at == "next_open"
              else _on(ex, "close", idx2, False))
    for col, (d, px_) in delist_exits(o, c).items():                     # 상장폐지 강제 청산
        t = d + CLOSE_T
        X.at[t, col] = True
        HALT.at[t, col] = False
        price.at[t, col] = px_

    # 최대 보유 봉수. 시가봉 진입(next_open): n일째 종가 = 2n-1, n일 다음날 시가 = 2n.
    # 종가봉 진입(close): n일 뒤 종가 = 2n, n일 뒤 시가 = 2n-1. 장중봉 진입(3봉/일): n일째 종가 = 3n-2, 다음날 시가 = 3n-1.
    hold = -1
    if spec.hold:
        op = spec.exit_hold_at == "open"
        if spec.entry_at == "close":
            hold = 2 * spec.hold - (1 if op else 0)
        elif intra:
            hold = 3 * spec.hold - 2 + (1 if op else 0)
        else:
            hold = 2 * spec.hold - 1 + (1 if op else 0)

    def at_entry(v):  # 손절·익절 값: 진입 봉 시점에 그 값이 있어야 한다(신호일 정보)
        if v is None:
            return np.nan
        if np.isscalar(v):
            return v
        v = v.reindex(columns=cols)
        g = (_on(v.shift(1), "open", idx2) if spec.entry_at == "next_open"
             else _on(v, "mid" if intra else "close", idx2))
        return g.ffill()

    fee_col = fees.reindex(idx).to_numpy()
    F = np.repeat(fee_col, 3 if intra else 2)[:, None]                      # 날짜별, 그날 봉 모두 동일
    if isinstance(slip, tuple):   # (시가봉, 종가봉[, 장중봉]) 슬리피지를 따로 — 동시호가/연속매매 구분용
        so, sc = (x.reindex(columns=cols) for x in slip[:2])
        S = _on(so, "open", idx2).combine_first(_on(sc, "close", idx2))
        if intra:
            S = S.combine_first(_on((slip[2] if len(slip) > 2 else sc).reindex(columns=cols), "mid", idx2))
        S = S.ffill().fillna(0.005)
    else:
        sl_ = slip.reindex(columns=cols)
        S = _on(sl_, "open", idx2).combine_first(_on(sl_, "close", idx2)).ffill().fillna(0.005)

    entry_i = np.full(len(cols), -1, dtype=np.int64)
    pending = np.zeros(len(cols), dtype=np.bool_)
    return vbt.Portfolio.from_signals(
        C, signal_func_nb=_signal_func,
        signal_args=(E.to_numpy(), X.to_numpy(), hold, entry_i, HALT.to_numpy(), pending),
        price=price, open=O, high=H, low=L,
        sl_stop=at_entry(spec.sl), tp_stop=at_entry(spec.tp), sl_trail=spec.trail,
        stop_entry_price="fillprice", stop_exit_price="stopmarket",    # 손절 기준 = 체결가, 손절 체결에도 슬리피지
        fees=F, slippage=S, size=np.inf, direction="longonly",
        init_cash=1.0, freq="12h",
    )


def allocate(pf, spec: Spec, trdval: pd.DataFrame) -> pd.DataFrame:
    """동시 보유 최대 N 을 지키는 거래만 채택(진입 시각 순, 같으면 신호일 우선순위 큰 순)."""
    tr = pf.trades.records_readable
    if tr.empty:
        return tr
    tr = tr.rename(columns={"Column": "col", "Entry Timestamp": "t_in", "Exit Timestamp": "t_out"})
    pri = (spec.priority if spec.priority is not None else trdval).reindex(columns=pf.wrapper.columns)
    day = tr["t_in"].dt.normalize()
    # 우선순위는 진입 판단 시점에 알 수 있는 값만: 시가·장중 진입은 전일, 종가 진입은 당일(15:20 근사)
    ri = pri.index.get_indexer(day) - (0 if spec.entry_at == "close" else 1)
    ci = pri.columns.get_indexer(tr["col"])
    vals = pri.to_numpy()
    tr["pri"] = np.where(ri >= 0, vals[np.clip(ri, 0, None), ci], 0.0)
    tr = tr.sort_values(["t_in", "pri"], ascending=[True, False])
    open_until, keep = [], []
    for t_in, t_out in zip(tr["t_in"], tr["t_out"]):
        open_until = [x for x in open_until if x > t_in]   # 같은 시각 청산분은 먼저 비운다
        ok = len(open_until) < spec.slots
        if ok:
            open_until.append(t_out)
        keep.append(ok)
    return tr[keep]


def portfolio_returns(pf, trades: pd.DataFrame, slots: int) -> pd.Series:
    """채택 거래의 종목별 수익률(vectorbt)을 거래 구간에만 합산 → 1/N 동일비중 일간 수익률."""
    r = pf.returns()
    idx2, cols = r.index, r.columns
    m = np.zeros((len(idx2) + 1, len(cols)), dtype=np.int32)
    a = idx2.get_indexer(trades["t_in"])
    b = idx2.get_indexer(trades["t_out"])
    ci = cols.get_indexer(trades["col"])
    np.add.at(m, (a, ci), 1)
    np.add.at(m, (b + 1, ci), -1)
    mask = np.cumsum(m, axis=0)[:-1] > 0
    port = pd.Series(np.where(mask, r.to_numpy(), 0.0).sum(axis=1) / slots, index=idx2)
    return (1 + port).groupby(port.index.normalize()).prod() - 1


def metrics(daily: pd.Series, trades: pd.DataFrame = None, slots: int = 10) -> dict:
    if daily.std() == 0 or daily.isna().all():  # 거래 없음
        return {"CAGR": 0.0, "Trades": 0}
    ra = daily.vbt.returns(freq="1D")
    ex = daily - rf_daily(daily.index)            # 샤프·소르티노는 단기채 대비 초과수익 기준(2026-09-25)
    dn = np.sqrt((ex.clip(upper=0) ** 2).mean())
    out = {
        "CAGR": ra.annualized(), "Vol": ra.annualized_volatility(),
        "Sharpe": ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
        "Sortino": ex.mean() / dn * np.sqrt(252) if dn > 0 else np.nan,
        "Sharpe_raw": ra.sharpe_ratio(),
        "MDD": ra.max_drawdown(), "Calmar": ra.calmar_ratio(),
        "Exposure": float((daily != 0).mean()),
    }
    if trades is None:  # 리밸런싱형: 거래 대신 월간 수익률로 승률·손익비·PF
        mo = (1 + daily).resample("ME").prod() - 1
        win, loss = mo[mo > 0], mo[mo <= 0]
        out.update({
            "Trades": len(mo), "WinRate": len(win) / len(mo) if len(mo) else np.nan,
            "Payoff": win.mean() / -loss.mean() if len(win) and len(loss) and loss.mean() < 0 else np.nan,
            "PF": win.sum() / -loss.sum() if loss.sum() < 0 else np.nan,
            "AvgTrade": mo.mean(),
        })
    elif len(trades):
        pnl = trades["Return"]
        win, loss = pnl[pnl > 0], pnl[pnl <= 0]
        yrs = max((daily.index[-1] - daily.index[0]).days / 365.25, 1e-9)
        out.update({
            "Trades": len(trades), "WinRate": len(win) / len(pnl),
            "Payoff": win.mean() / -loss.mean() if len(win) and len(loss) and loss.mean() < 0 else np.nan,
            "PF": win.sum() / -loss.sum() if loss.sum() < 0 else np.nan,
            "AvgTrade": pnl.mean(),
            "AvgHoldDays": ((trades["t_out"] - trades["t_in"]).dt.total_seconds() / 86400).mean(),
            "Turnover": 2 * len(trades) / slots / yrs,
        })
    return out


def rebalance(weights: pd.DataFrame, px: dict, fees: pd.Series, slip: pd.DataFrame):
    """리밸런싱형: 신호일 목표비중(행=리밸런싱일, 합≤1) → 다음 거래일 시가에 TargetPercent 주문."""
    cols = weights.columns[(weights.fillna(0) > 0).any()]
    o = px["open"].reindex(columns=cols)
    c = px["close"].reindex(columns=cols)
    w = weights[cols].reindex(c.index)
    rows = w.notna().any(axis=1)
    w = w.where(~rows, w.fillna(0.0)).shift(1)       # 리밸런싱일엔 비대상=0, t 신호 → t+1 체결
    price = o.copy()
    # 정지로 못 낸 주문은 정지가 풀린 첫 거래일로 미룬다(다음 리밸런싱 전까지)
    rb = w.notna().any(axis=1).to_numpy()
    for j, col in enumerate(cols):
        oj, wj = o[col].to_numpy(), w[col].to_numpy().copy()
        k = 0
        while k < len(wj):
            if not np.isnan(wj[k]) and np.isnan(oj[k]):
                m = k + 1
                while m < len(wj) and np.isnan(oj[m]) and not rb[m]:
                    m += 1
                if m < len(wj) and not rb[m]:
                    wj[m] = wj[k]
                wj[k] = np.nan
                k = m
            else:
                k += 1
        w[col] = wj
    for col, (d, px_) in delist_exits(o, c).items():   # 상장폐지: 청산가로 전량 매도해 현금화
        w.at[d, col] = 0.0
        price.at[d, col] = px_
    fee_arr = (fees.reindex(index=c.index, columns=cols).to_numpy() if isinstance(fees, pd.DataFrame)
               else fees.reindex(c.index).to_numpy()[:, None])
    pf = vbt.Portfolio.from_orders(
        c.ffill(), size=w, size_type="targetpercent", price=price.fillna(c).ffill(),
        fees=fee_arr,
        slippage=slip.reindex(index=c.index, columns=cols).ffill().fillna(0.005),
        group_by=True, cash_sharing=True, call_seq="auto", init_cash=1.0, freq="1D",
        direction="longonly",
    )
    return pf


def rebalance_turnover(pf) -> float:
    """연 회전율 = 연간 매도금액 / 평균 평가액."""
    orders = pf.orders.records_readable
    sells = orders[orders["Side"] == "Sell"]
    val = pf.value()
    yrs = (val.index[-1] - val.index[0]).days / 365.25
    return float((sells["Size"] * sells["Price"]).sum() / val.mean() / yrs) if len(sells) else 0.0
