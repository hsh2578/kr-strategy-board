"""2차 조사 전략(2026-09-23): 스윙·깃허브 코드·중장기·자산배분. strategies.py 끝에서 import 되어 REG 에 등록된다.

자산배분의 Keller 계열(HAA·VAA·DAA·LAA)은 신호를 미국 ETF/실업률로 계산하고 매매는 국내 상장 ETF로 한다.
"""
import numpy as np
import pandas as pd

import kr_data as K
from engine import Spec
from strategies import (RB, _me, _qe, atr, d, etf_px, fund, hh, ll, ma, month_ends, pick, rsi, stock, strat)


# ─── 공용 ───
def _large(n=200):
    """유니버스 내 시총 상위 n (대형주)."""
    return d.CAP.where(d.U).rank(axis=1, ascending=False) <= n


def _etf_signal(code, entry, exit=None, entry_at="close", exit_at="close", hold=None, exit_hold_at="close"):
    return Spec(entry=entry.fillna(False), entry_at=entry_at, exit=exit, exit_at=exit_at, hold=hold,
                exit_hold_at=exit_hold_at, slots=1, etf=True, meta={"px": (code,)})


def _cohort(sel, k):
    """매월 선택(행 합 1)을 k개월 겹쳐 보유 → k개 코호트 평균 비중."""
    return sel.rolling(k, min_periods=1).mean()


def _me_px(px: pd.DataFrame) -> pd.DataFrame:
    return px.groupby(px.index.to_period("M")).last()


def _ret_m(mpx, k):
    return mpx / mpx.shift(k) - 1


def m13612w(mpx):
    return 12 * _ret_m(mpx, 1) + 4 * _ret_m(mpx, 3) + 2 * _ret_m(mpx, 6) + _ret_m(mpx, 12)


def m13612u(mpx):
    return (_ret_m(mpx, 1) + _ret_m(mpx, 3) + _ret_m(mpx, 6) + _ret_m(mpx, 12)) / 4


def avg_mom_score(mpx):
    """1~12개월 각각 현재가>과거가면 1 → 평균(0~1)."""
    score = sum((mpx > mpx.shift(k)).astype(float) for k in range(1, 13)) / 12
    return score.where(mpx.shift(12).notna())   # 12개월 이력 전엔 점수 없음(과소평가 방지)


def _etf_alloc(codes, fn, freq="M", start=None):
    """월말(또는 분기·연)마다 fn(t) → {code: weight}. 상장 전 ETF 는 비중 0 으로 둔다."""
    p = etf_px(*codes)
    idx = p["close"].dropna(how="all").index
    dts = month_ends(idx)
    if freq == "Q":
        dts = dts[dts.month.isin([3, 6, 9, 12])]
    elif freq == "A":
        dts = dts[dts.month == 12]
    if start is not None:
        dts = dts[dts >= pd.Timestamp(start)]
    rows = {}
    for t in dts:
        w = fn(t) or {}
        rows[t] = {c: w.get(c, 0.0) for c in codes}
    return RB(pd.DataFrame(rows).T, etf=True, px=p)


def _kr_me(*codes):
    return _me_px(etf_px(*codes)["close"])


def _us_me():
    return _me_px(K.load("us"))


def _at(frame, t):
    """월말 프레임에서 t 가 속한 달의 행."""
    p = pd.Timestamp(t).to_period("M")
    return frame.loc[p] if p in frame.index else None


# ═══════════════ 스윙 ═══════════════
def _flow_spike(hold):
    ind = -(d.FOR + d.INST)                                   # 개인 ≈ -(외국인+기관)
    big = np.maximum(d.FOR, d.INST)
    strength = big * 1e6 / d.CAP                              # 순매수(백만원) / 시총
    spike = (d.V > 5 * ma(d.V, 20).shift(1)) & (big > ind) & (big > 0) & (K.load("close") >= 1000) & d.U
    ev = strength.where(spike).stack().sort_index(level=0)
    # 강도 상위 25% 경계는 그 이전 이벤트 분포로만(사후 경계 금지)
    thr = ev.expanding(min_periods=50).quantile(0.75).groupby(level=0).last().shift(1)
    q4 = strength.ge(thr.reindex(strength.index), axis=0) & spike
    cool = q4.rolling(50, min_periods=1).sum().shift(1).fillna(0) == 0
    return stock(q4 & cool, hold=hold)


@strat("SW15", "스윙", "기관·외인 주도 거래량 5배 폭증(강도 상위25%) → 20일 보유", "Kang, arXiv 2512.14134 (한국 2020~24)",
       assume="개인 = -(외국인+기관). 강도 경계는 과거 이벤트 누적 75% 분위")
def sw15(): return _flow_spike(20)


@strat("SW16", "스윙", "기관·외인 주도 거래량 5배 폭증(강도 상위25%) → 50일 보유", "Kang, arXiv 2512.14134 (한국 2020~24)",
       assume="SW15 와 동일, 보유 50일")
def sw16(): return _flow_spike(50)


def _tom(code):
    c = etf_px(code)["close"]
    pos_from_end = pd.Series(1, index=c.index).groupby(c.index.to_period("M")).cumcount(ascending=False)
    e = pd.DataFrame((pos_from_end == 1).to_numpy()[:, None], index=c.index, columns=c.columns)
    return _etf_signal(code, e, hold=4)


@strat("SW17", "스윙", "월말·월초(TOM): 코스닥150 ETF 월말 전날 종가→다음달 3일째 종가", "JDQS 2022 코스닥 TOM", etf=True)
def sw17(): return _tom("229200")


@strat("SW18", "스윙", "월말·월초(TOM): KODEX200 월말 전날 종가→다음달 3일째 종가", "JDQS 2022, QuantConnect #35", etf=True)
def sw18(): return _tom("069500")


def _double7(code):
    c = etf_px(code)["close"]
    return _etf_signal(code, (c > ma(c, 200)) & (c <= ll(c, 7)), exit=c >= hh(c, 7))


@strat("SW19", "스윙", "Double 7s KODEX200(200일선 위 7일 최저 종가매수 / 7일 최고 매도)", "Connors, The Robust Trader", etf=True)
def sw19(): return _double7("069500")


@strat("SW20", "스윙", "Double 7s 코스닥150 ETF", "Connors, The Robust Trader", etf=True)
def sw20(): return _double7("229200")


@strat("SW21", "스윙", "Connors %b KODEX200(%b 3일 연속<0.2 매수 / >0.8 매도)", "Connors %b", etf=True)
def sw21():
    c = etf_px("069500")["close"]
    m, s = ma(c, 20), c.rolling(20).std()
    b = (c - (m - 2 * s)) / (4 * s)
    e = (c > ma(c, 200)) & (b < 0.2) & (b.shift(1) < 0.2) & (b.shift(2) < 0.2)
    return _etf_signal("069500", e, exit=b > 0.8)


@strat("SW22", "스윙", "누적 RSI(2)<10 대형주(200일선 위) → 누적>65 청산, 최대 3종목", "Quantitativo cumulative RSI",
       assume="대형주 = 유니버스 시총 상위 200")
def sw22():
    r2 = rsi(d.C, 2)
    cum = r2 + r2.shift(1)
    m200 = ma(d.C, 200)
    return stock((d.C > m200) & (cum < 10) & _large(), exit=(cum > 65) | (d.C < m200), slots=3)


@strat("SW23", "스윙", "Connors R3 대형주(RSI2 3일 연속 하락·당일<10) → RSI2>70 청산", "Connors R3 (WH SelfInvest)",
       assume="대형주 = 시총 상위 200, 종가 진입")
def sw23():
    r2 = rsi(d.C, 2)
    e = ((d.C > ma(d.C, 200)) & (r2 < r2.shift(1)) & (r2.shift(1) < r2.shift(2)) & (r2.shift(2) < 60)
         & (r2 < 10) & _large())
    return stock(e, entry_at="close", exit=r2 > 70, exit_at="close")


@strat("SW24", "스윙", "오닐 베이스 돌파(60일 박스 15% 이내)+거래량 2.5배 → 63일 보유", "O'Neil Global Advisors 2022",
       assume="베이스 = 직전 60일 고저폭 15% 이내 박스로 근사")
def sw24():
    top, bot = hh(d.H, 60).shift(1), ll(d.L, 60).shift(1)
    return stock(((top / bot - 1) <= 0.15) & (d.H > top) & (d.V >= 2.5 * ma(d.V, 50).shift(1)), hold=63)


@strat("SW25", "스윙", "거래량 급감(5일<50일의 50%) 상승추세 → 20일 보유", "Chae & Kang 2019 한국 저거래량 프리미엄",
       assume="원문 규칙 미확인 → 조사 에이전트 제안 규칙, 조건 최초 성립일만")
def sw25():
    e = (d.C > ma(d.C, 20)) & (ma(d.V, 5) < 0.5 * ma(d.V, 50)) & (d.L > ll(d.L, 20).shift(1))
    return stock(e & ~e.shift(1, fill_value=False), hold=20)


# ─── 깃허브 코드 이식 ───
@strat("SW26", "스윙", "KIS 공식 trend_filter(종가>SMA60 & 전일比 상승, 손절5%·익절10%)",
       "koreainvestment/open-trading-api preset trend_filter_signal")
def sw26():
    on = (d.C > ma(d.C, 60)) & (d.C > d.C.shift(1))
    off = (d.C < ma(d.C, 60)) & (d.C < d.C.shift(1))
    return stock(on & ~on.shift(1, fill_value=False), exit=off, sl=0.05, tp=0.10)


@strat("SW27", "스윙", "KIS 공식 momentum(ROC60 0 상향 돌파 / 0 하향 청산, 손절10%)",
       "koreainvestment/open-trading-api preset momentum")
def sw27():
    roc = d.C / d.C.shift(60) - 1
    return stock((roc > 0) & (roc.shift(1) <= 0), exit=roc < 0, sl=0.10)


@strat("SW28", "스윙", "KIS 공식 변동성 수축→확장(ATR10<ATR 20일평균 & 당일 +3%)",
       "koreainvestment/open-trading-api preset volatility_breakout")
def sw28():
    a = atr(d.H, d.L, d.C, 10)
    ret = d.C / d.C.shift(1) - 1
    return stock((a < ma(a, 20)) & (ret > 0.03), exit=ret < -0.03, sl=0.05)


@strat("SW29", "스윙", "KIS 공식 이격 역추세(종가<SMA20×0.9 매수 / >SMA20×1.1 매도)",
       "koreainvestment/open-trading-api preset ma_divergence")
def sw29():
    m = ma(d.C, 20)
    return stock(d.C < m * 0.9, exit=d.C > m * 1.1)


@strat("SW30", "스윙", "전고점(최대 10년) 신고가 매수 + 1×ATR10 트레일링 스톱", "awesome-systematic-trading trend-following-effect",
       assume="데이터가 2010년부터라 10년 미만 구간은 가용 기간 최고가")
def sw30():
    top = d.C.rolling(2520, min_periods=250).max().shift(1)
    return stock(d.C > top, sl=atr(d.H, d.L, d.C, 10) / d.C, trail=True)


# ═══════════════ 중장기 ═══════════════
@strat("LT29", "중장기", "대형주(시총 상위200) 12개월 모멘텀 상위 40, 월간", "키움증권 2016 / 인텔리퀀트 #441")
def lt29():
    dt = _me()
    return RB(pick(d.C / d.C.shift(252) - 1, _large(), 40, dt))


@strat("LT30", "중장기", "대형주 12-1 모멘텀 상위 80 중 FIP(꾸준한 상승) 상위 30, 월간", "Da·Gurun·Warachka 2014 RFS")
def lt30():
    dt = _me()
    mom = d.C.shift(21) / d.C.shift(252) - 1
    up = (d.R > 0).astype(float).rolling(231).mean().shift(21)
    dn = (d.R < 0).astype(float).rolling(231).mean().shift(21)
    ids = np.sign(mom) * (dn - up)
    top = mom.where(_large()).rank(axis=1, ascending=False) <= 80
    return RB(pick(-ids, top, 30, dt))


@strat("LT31", "중장기", "Clenow Stocks on the Move 20종목, 주간(지수 200일선 아래면 현금)", "Clenow 『Stocks on the Move』",
       assume="ATR 사이징 대신 동일비중. 약세장엔 전량 현금(원전은 신규매수만 중단)")
def lt31():
    y = np.log(d.C)
    n = 90
    x = np.arange(n) - (n - 1) / 2
    wk = pd.DatetimeIndex(d.C.index.to_series().groupby(d.C.index.to_period("W")).last())
    kodex = etf_px("069500")["close"]["069500"]
    bull = (kodex > ma(kodex, 200)).reindex(wk).fillna(False)
    gap_ok = d.R.abs().rolling(90).max() < 0.15
    rows = {}
    for t in wk:
        i = y.index.get_loc(t)
        if i < n:
            continue
        Y = y.iloc[i - n + 1:i + 1]
        Yc = Y - Y.mean()
        slope = Yc.mul(x, axis=0).sum() / (x ** 2).sum()
        r2 = 1 - ((Yc - np.outer(x, slope)) ** 2).sum() / (Yc ** 2).sum()
        rows[t] = ((np.exp(slope * 250) - 1) * r2).where(Y.notna().all())
    score = pd.DataFrame(rows).T.reindex(wk)
    w = pick(score, d.U & (d.C > ma(d.C, 100)) & gap_ok, 20, wk)
    w.loc[~bull.to_numpy()] = 0.0
    return RB(w)


@strat("LT32", "중장기", "연말 배당 교대(9월말 대형 고배당30 → 12월말 전 매도·코스닥 12월 낙폭과대30 → 1/15 매도)",
       "한투·KB증권 리포트, 한국거래소 1월 효과 집계", assume="배당락 전 매도 = 12월 마지막 4번째 거래일 신호")
def lt32():
    idx = d.C.index
    s = idx.to_series()
    kq = pd.Series(d.C.columns.map(d.META["market"]).str.contains("KOSDAQ"), index=d.C.columns)
    div = fund("DIV")
    rows = []
    for y in sorted(set(idx.year)):
        sep = s[(s.dt.year == y) & (s.dt.month == 9)]
        dec = s[(s.dt.year == y) & (s.dt.month == 12)]
        jan = s[(s.dt.year == y + 1) & (s.dt.month == 1) & (s.dt.day >= 15)]
        if len(sep) == 0 or len(dec) < 4 or len(jan) == 0:
            continue
        t1, t2, t3 = sep.iloc[-1], dec.iloc[-4], jan.iloc[0]
        w1 = pick(div.reindex([t1]), _large().reindex([t1]), 30, pd.DatetimeIndex([t1]))
        drop = (d.C.loc[t2] / d.C.loc[:t2].iloc[-15] - 1).where(d.U.loc[t2] & kq)
        sel = drop.rank() <= 30
        w2 = (sel / sel.sum()).fillna(0.0).to_frame(t2).T
        rows += [w1, w2, pd.DataFrame(0.0, index=[t3], columns=d.C.columns)]
    return RB(pd.concat(rows).reindex(columns=d.C.columns).fillna(0.0))


@strat("LT33", "중장기", "easygap 상대강도 로테이션(0.6·60일+0.4·120일 > 0, SMA60 위) 상위10, 월간",
       "easygap/quant_trader relative_strength_rotation",
       assume="원문 기본값대로 지수 200일선 필터 없음. 유니버스는 원문 전종목 대신 시총 1,000억 이상, 사이징은 1% 리스크 대신 동일비중. "
              "익절 7%·SMA60 이탈 매도는 LT33b")
def lt33():
    dt = _me()
    score = 0.6 * (d.C / d.C.shift(60) - 1) + 0.4 * (d.C / d.C.shift(120) - 1)
    ok = d.U & (score > 0) & (d.C > ma(d.C, 60))
    return RB(pick(score, ok, 10, dt))


@strat("LT34", "중장기", "시장상태 모멘텀(KODEX200 12개월>0이면 6개월 모멘텀 상위20, 아니면 국고채10년)",
       "QuantConnect Strategy Library #37 (long-only 변형)")
def lt34():
    from strategies import _with_filter  # noqa
    dt = _me()
    kodex = etf_px("069500")["close"]["069500"]
    up = (kodex / kodex.shift(252) - 1 > 0).reindex(dt).fillna(False)
    w = pick(d.C / d.C.shift(126) - 1, d.U, 20, dt)
    w.loc[~up.to_numpy()] = 0.0
    w["148070"] = np.where(up, 0.0, 1.0)
    from strategies import mixed_px
    return RB(w, px=mixed_px("148070"))


@strat("LT35", "중장기", "Consistent momentum(두 6개월 구간 모두 상위10%, 1개월 대기, 6개월 코호트)",
       "awesome-systematic-trading consistent-momentum")
def lt35():
    dt = _me()
    a = (d.C.shift(21) / d.C.shift(147) - 1).reindex(dt)
    b = (d.C / d.C.shift(126) - 1).reindex(dt)
    u = d.U.reindex(dt)
    top = lambda x: x.where(u).rank(axis=1, pct=True) >= 0.9
    sel = (top(a) & top(b)).shift(1, fill_value=False)       # 1개월 대기
    sel = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return RB(_cohort(sel, 6))


@strat("LT36", "중장기", "12개월 주기(1년 전 같은 달 수익률 상위 30), 월간", "Heston-Sadka, awesome-systematic-trading 12-month-cycle")
def lt36():
    dt = _me()
    mpx = d.C.reindex(dt)
    past = (mpx.shift(11) / mpx.shift(12) - 1)
    return RB(pick(past, d.U, 30, dt))


@strat("LT37", "중장기", "고변동성 내 모멘텀(시총 상위50%·변동성 상위20% 중 6개월 상위20%, 6개월 코호트)",
       "QuantConnect Strategy Library #155")
def lt37():
    dt = _me()
    cap = d.CAP.where(d.U).reindex(dt)
    vol = d.R.rolling(252, min_periods=200).std().reindex(dt)
    mom = (d.C.shift(5) / d.C.shift(126) - 1).reindex(dt)
    big = cap.rank(axis=1, pct=True) >= 0.5
    hv = vol.where(big).rank(axis=1, pct=True) >= 0.8
    sel = mom.where(hv).rank(axis=1, pct=True) >= 0.8
    sel = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return RB(_cohort(sel, 6))


@strat("LT38", "중장기", "모멘텀+회전율(12개월 상위20% 중 회전율 상위30, 3개월 코호트)", "QuantConnect Strategy Library #66",
       assume="회전율 = 21일 평균 거래대금/시총, 원전 상위1% 대신 상위 30종목")
def lt38():
    dt = _me()
    mom = (d.C / d.C.shift(252) - 1).where(d.U).reindex(dt)
    turn = (d.TV / d.CAP).rolling(21, min_periods=15).mean().reindex(dt)
    top = mom.rank(axis=1, pct=True) >= 0.8
    sel = turn.where(top).rank(axis=1, ascending=False) <= 30
    sel = sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return RB(_cohort(sel, 3))


# ═══════════════ 자산배분 (국내 ETF 매매) ═══════════════
@strat("AA07", "자산배분", "코스피 절대모멘텀(KODEX200 10개월선 위 → 주식, 아래 → 국고채3년)", "강환국 『거인의 포트폴리오』", etf=True)
def aa07():
    m = _kr_me("069500")["069500"]
    ok = m > m.rolling(10).mean()
    return _etf_alloc(["069500", "114260"],
                      lambda t: {"069500": 1.0} if bool(_at(ok, t)) else {"114260": 1.0})


@strat("AA08", "자산배분", "systrader79 평균모멘텀 스코어(주식:채권:현금 = 점수:점수:1)", "systrader79 『주식투자 ETF로 시작하라』", etf=True)
def aa08():
    s = avg_mom_score(_kr_me("069500", "148070"))
    def fn(t):
        r = _at(s, t)
        if r is None or r.isna().any():
            return {}
        tot = r["069500"] + r["148070"] + 1
        return {"069500": r["069500"] / tot, "148070": r["148070"] / tot, "153130": 1 / tot}
    return _etf_alloc(["069500", "148070", "153130"], fn)


@strat("AA09", "자산배분", "HAA-Simple(미국 TIP·SPY 모멘텀>0 → S&P500, 아니면 국고채10년/단기채 중 강한 쪽)",
       "Keller & Keuning 2023 HAA", etf=True, assume="신호는 미국 TIP·SPY·IEF·BIL, 매매는 143850·148070·153130")
def aa09():
    m = m13612u(_us_me())
    def fn(t):
        r = _at(m, t)
        if r is None or r[["TIP", "SPY", "IEF", "BIL"]].isna().any():
            return {}
        if r["TIP"] > 0 and r["SPY"] > 0:
            return {"143850": 1.0}
        return {"148070": 1.0} if r["IEF"] > r["BIL"] else {"153130": 1.0}
    return _etf_alloc(["143850", "148070", "153130"], fn)


@strat("AA10", "자산배분", "LAA 한국판(KODEX200·금·국고채10년 각25% + 나스닥 25%↔단기채)", "Keller LAA, 거인의 포트폴리오", etf=True,
       assume="미국가치주(IWD) 대신 KODEX200. 약세 = SPY<200일선 AND 실업률>12개월평균")
def aa10():
    spy = K.load("us")["SPY"]
    bear_px = _me_px((spy < ma(spy, 200)).to_frame())["SPY"]
    un = K.load("unrate").iloc[:, 0]
    un_m = un.groupby(un.index.to_period("M")).last()
    bad_un = (un_m > un_m.rolling(12).mean()).shift(1)        # 실업률은 다음 달 발표 → 1개월 지연
    def fn(t):
        p = pd.Timestamp(t).to_period("M")
        risk_off = bool(bear_px.get(p, False)) and bool(bad_un.get(p, False))
        return {"069500": .25, "132030": .25, "148070": .25, ("153130" if risk_off else "133690"): .25}
    return _etf_alloc(["069500", "132030", "148070", "133690", "153130"], fn)


@strat("AA11", "자산배분", "DAA1-U1(미국 VWO·BND 카나리아 하나라도 음수 → 방어 100%, 아니면 S&P500)",
       "Keller & Keuning DAA 2018", etf=True,
       assume="원문 B=1(카나리아 1개 불량 시 전액 방어). 방어 후보는 원문 SHV·IEF·UST 대신 IEF vs SHY 13612W, 공격은 S&P500 1개(U1)")
def aa11():
    m = m13612w(_us_me())
    def fn(t):
        r = _at(m, t)
        if r is None or r[["VWO", "BND", "IEF", "SHY"]].isna().any():
            return {}
        if r["VWO"] > 0 and r["BND"] > 0:
            return {"143850": 1.0}
        return {"148070": 1.0} if r["IEF"] > r["SHY"] else {"153130": 1.0}
    return _etf_alloc(["143850", "148070", "153130"], fn)


@strat("AA12", "자산배분", "한미 평균모멘텀 스코어(한국주식·미국주식·국고채10년·미국장기채 각 25%×점수, 나머지 단기채)",
       "강환국 『거인의 포트폴리오』", etf=True, assume="미국장기채 304660 상장(2018-09) 전 몫은 단기채")
def aa12():
    codes = ["069500", "143850", "148070", "304660"]
    s = avg_mom_score(_kr_me(*codes))
    def fn(t):
        r = _at(s, t)
        if r is None:
            return {}
        w = {c: 0.25 * float(r[c]) for c in codes if pd.notna(r[c])}
        w["153130"] = 1 - sum(w.values())
        return w
    return _etf_alloc(codes + ["153130"], fn)


@strat("AA13", "자산배분", "GTAA5 한국판(미국주식·선진국·국고채10년·금·리츠 각20%, 10개월선 아래면 단기채)",
       "Faber 2013 개정판", etf=True, assume="원자재(DBC) 대신 금")
def aa13():
    codes = ["143850", "195970", "148070", "132030", "182480"]
    m = _kr_me(*codes)
    ok = m > m.rolling(10).mean()
    def fn(t):
        r = _at(ok, t)
        if r is None:
            return {}
        w = {c: 0.2 for c in codes if bool(r[c])}
        w["153130"] = 1 - sum(w.values())
        return w
    return _etf_alloc(codes + ["153130"], fn)


@strat("AA14", "자산배분", "HAA-Balanced 7자산(미국 TIP 카나리아, 13612U 상위4, 음수 몫은 미국채10년/달러단기채)",
       "Keller & Keuning 2023 HAA", etf=True, assume="원자재 제외 7자산. 순위는 국내 ETF 가격 기준")
def aa14():
    off = ["143850", "280930", "195970", "195980", "182480", "305080", "304660"]
    dfn = ["305080", "329750"]
    mk = m13612u(_kr_me(*(off + dfn)))
    tip = m13612u(_us_me())["TIP"]
    def fn(t):
        r, c = _at(mk, t), _at(tip, t)
        if r is None or c is None or pd.isna(c):
            return {}
        best_def = r[dfn].dropna().idxmax() if r[dfn].notna().any() else "305080"
        if c <= 0:
            return {best_def: 1.0}
        top = r[off].dropna().nlargest(4)
        w = {}
        for code, v in top.items():
            k = code if v > 0 else best_def
            w[k] = w.get(k, 0) + 0.25
        return w
    return _etf_alloc(off + dfn, fn)


@strat("AA15", "자산배분", "VAA 공격형(미국 SPY·EFA·EEM·AGG 모두 양수면 최강 1개, 아니면 방어 최강 1개)",
       "Keller & Keuning 2017 VAA", etf=True,
       assume="신호=미국, 매매=143850·195970·195980·273130 / 방어 136340·148070·153130")
def aa15():
    m = m13612w(_us_me())
    off = {"SPY": "143850", "EFA": "195970", "EEM": "195980", "AGG": "273130"}
    dfn = {"LQD": "136340", "IEF": "148070", "SHY": "153130"}
    def fn(t):
        r = _at(m, t)
        if r is None or r[list(off) + list(dfn)].isna().any():
            return {}
        if (r[list(off)] > 0).all():
            return {off[r[list(off)].idxmax()]: 1.0}
        return {dfn[r[list(dfn)].idxmax()]: 1.0}
    return _etf_alloc(list(off.values()) + list(dfn.values()), fn)


@strat("AA16", "자산배분", "섹터 ETF 모멘텀(섹터 9개 중 12개월 수익률 상위3 동일비중), 월간",
       "QuantConnect #14 Sector Momentum(원문: 12개월·상위3·필터 없음)", etf=True,
       assume="원문대로 1개월 건너뛰기·절대모멘텀 필터 없음(이전 12-1+필터 버전은 여러 변형 중 최고 성과라 사후 선택 의심으로 폐기). "
              "섹터 목록은 TIGER200 섹터 8개 + TIGER 헬스케어(143860)")
def aa16():
    sec = K.SECTOR_ETFS
    m = _kr_me(*sec)
    mom = m / m.shift(12) - 1
    def fn(t):
        r = _at(mom, t)
        if r is None or r.notna().sum() < 5:
            return {}
        return {c: 1 / 3 for c in r.dropna().nlargest(3).index}
    return _etf_alloc(sec + ["153130"], fn)


@strat("AA17", "자산배분", "코스피·나스닥100·달러 1:1:1, 분기", "원달러 역상관 활용(네이버 블로그)", etf=True)
def aa17():
    return _etf_alloc(["069500", "133690", "138230"], lambda t: {"069500": 1 / 3, "133690": 1 / 3, "138230": 1 / 3}, "Q")


@strat("AA18", "자산배분", "Paired switching(KODEX200 vs 국고채10년 직전 분기 수익 높은 쪽 100%)", "QuantConnect Strategy Library #33", etf=True)
def aa18():
    m = _kr_me("069500", "148070")
    r3 = m / m.shift(3) - 1
    def fn(t):
        r = _at(r3, t)
        if r is None or r.isna().any():
            return {}
        return {r.idxmax(): 1.0}
    return _etf_alloc(["069500", "148070"], fn, "Q")


@strat("AA19", "자산배분", "골든 버터플라이 한국판(KODEX200·200중소형·국고채10년·단기채·금 각 20%), 연1회",
       "Portfolio Charts Golden Butterfly", etf=True, assume="소형가치 대신 200중소형, 장기채 대신 국고채10년")
def aa19():
    codes = ["069500", "226980", "148070", "153130", "132030"]
    return _etf_alloc(codes, lambda t: dict.fromkeys(codes, 0.2), "A")


@strat("AA20", "자산배분", "자산군 모멘텀(미국주식·선진국·국고채10년·금·리츠 중 12개월 상위3), 월간", "QuantConnect Strategy Library #13", etf=True)
def aa20():
    codes = ["143850", "195970", "148070", "132030", "182480"]
    m = _kr_me(*codes)
    r12 = m / m.shift(12) - 1
    def fn(t):
        r = _at(r12, t)
        if r is None or r.notna().sum() < 3:
            return {}
        return {c: 1 / 3 for c in r.dropna().nlargest(3).index}
    return _etf_alloc(codes, fn)


@strat("LT33b", "중장기", "LT33 + 원문 청산(익절 7%, SMA60 하향 이탈 즉시 매도, 월말 탈락 매도)",
       "easygap/quant_trader relative_strength_rotation + config/strategies.yaml", assume="대형주 = 시총 상위 30, 신호형 10슬롯")
def lt33b():
    dt = _me()
    score = 0.6 * (d.C / d.C.shift(60) - 1) + 0.4 * (d.C / d.C.shift(120) - 1)
    m60 = ma(d.C, 60)
    ok = d.U & (score > 0) & (d.C > m60)
    sel = pick(score, ok, 10, dt).gt(0)
    entry = sel.reindex(d.C.index, fill_value=False)
    dropped = (~sel).reindex(d.C.index, fill_value=False)          # 월말에 선정 탈락
    exit_ = (d.C < m60) & (d.C.shift(1) >= m60.shift(1)) | dropped
    return Spec(entry=entry.fillna(False), exit=exit_.fillna(False), tp=0.07, slots=10)
