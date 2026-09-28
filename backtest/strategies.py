"""전략 정의. 각 함수는 engine.Spec(신호형) 또는 dict(kind='rebalance', weights=...)를 돌려준다.

규칙
- 신호는 t일 장 마감 후 알 수 있는 정보만 쓴다(rolling 은 당일까지, 돌파 기준선은 shift(1)).
- 개별종목은 보통주 + t일 시총 1,000억 이상(kr_data.universe) 안에서만 진입한다.
- 파라미터는 조사한 출처 값. 출처에 없어 정한 값은 meta['assume'] 에 적는다.
"""
import os

import numpy as np
import pandas as pd

import kr_data as K
from engine import Spec

REG = {}


def strat(sid, cat, name, src, assume="", etf=False, since=None):
    def deco(f):
        REG[sid] = dict(fn=f, cat=cat, name=name, src=src, assume=assume, etf=etf, since=since)
        return f
    return deco


# ─── 공용 데이터·지표 ───
class D:
    """지연 로드 캐시."""
    _c = {}

    def __getattr__(self, k):
        if k not in self._c:
            m = {"O": "adj_open", "H": "adj_high", "L": "adj_low", "C": "adj_close",
                 "V": "volume", "TV": "trdval", "CAP": "mktcap", "R": "ret"}
            if k in m:
                self._c[k] = K.load(m[k])
            elif k == "U":
                u = K.universe()
                if os.environ.get("BT_WARN", "") not in ("", "0"):   # 경고 필터 버전: 부실 전조 종목 매수 제외
                    u = u & ~warn_mask(u)
                self._c[k] = u
            elif k in ("FOR", "INST"):
                f = K.load("flow_foreign" if k == "FOR" else "flow_inst")
                self._c[k] = f.reindex(index=self.C.index, columns=self.C.columns)
            elif k == "META":
                self._c[k] = K.load("meta")
            else:
                raise AttributeError(k)
        return self._c[k]


d = D()


WARN_A = ("admin_issue", "unfaithful", "audit_opinion")     # 관리종목·불성실공시·감사의견
WARN_B = ("cb", "bw", "rights_offering")                    # 적자일 때만: CB·BW·유상증자


def warn_mask(like):
    """부실 전조(2026-09-24 사전 분석: 장기정지 후 부실 폐지 104건 중 87% 적중, 정상 종목 14% 제외).
    t일 기준 최근 250거래일 안에 A 공시가 있거나, TTM 순이익 적자이면서 B 공시가 있으면 True.
    공시 목록이 2015년부터라 2016년 이후에만 의미가 있다."""
    ev = pd.read_parquet(K.PANEL.parent / "dart" / "disclosures.parquet")
    idx, cols = like.index, like.columns
    pos = idx.searchsorted(pd.to_datetime(ev.rcept_dt), side="right") - 1
    ev = ev.assign(day=idx[pos.clip(0)])[pos >= 0]

    def recent(kinds):
        e = ev[ev.event.isin(kinds)]
        x = pd.crosstab(e.day, e.stock_code).gt(0).reindex(index=idx, columns=cols, fill_value=False)
        return x.astype(float).rolling(250, min_periods=1).max().gt(0)

    ni = pd.read_parquet(K.PANEL / "fin_ni_ttm.parquet").reindex(columns=cols)
    loss = ni.reindex(idx.union(ni.index)).ffill().reindex(idx).lt(0)
    return recent(WARN_A) | (loss & recent(WARN_B))


def ma(x, n): return x.rolling(n, min_periods=n).mean()
def hh(x, n): return x.rolling(n, min_periods=n).max()
def ll(x, n): return x.rolling(n, min_periods=n).min()


def rsi(c, n):
    diff = c.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def rsi_sma(c, n):
    """단순평균(SMA) RSI — swing-it 원문 방식(avg_gain/avg_loss 를 rolling mean)."""
    diff = c.diff()
    up = diff.clip(lower=0).rolling(n, min_periods=n).mean()
    dn = (-diff.clip(upper=0)).rolling(n, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    return tr.rolling(n, min_periods=n).mean()


def between(x, lo, hi): return (x >= lo) & (x <= hi)


def crossup(a, b): return (a > b) & (a.shift(1) <= b.shift(1))


def close_pos(h, l, c): return (c - l) / (h - l).replace(0, np.nan)


def top_trdval(n):
    """전일까지 거래대금 상위 n (유니버스 내)."""
    return d.TV.where(d.U).rank(axis=1, ascending=False) <= n


GENERIC_EXIT = dict(tp=0.10, sl=0.03, hold=5)   # 와치독 조건식 설정값(CD01 수급 조건식에만 남음)


def SHORT_EXIT():
    """원문에 청산이 없는 단기 조건식용(결과 보기 전 고정): 신호일 저가를 깨면 손절, 최대 3거래일."""
    return dict(sl=(1 - d.L / d.C).clip(lower=0.01), hold=3)


def stock(entry, **kw):
    return Spec(entry=(entry & d.U).fillna(False), **kw)


def month_ends(idx):
    s = idx.to_series()
    return pd.DatetimeIndex(s.groupby(idx.to_period("M")).last())


def pick(score, mask, n, dates, ascending=False):
    """리밸런싱일마다 score 상위(또는 하위) n 종목 동일비중. 행 = 리밸런싱일."""
    sc = score.reindex(dates).where(mask.reindex(dates).fillna(False))
    rk = sc.rank(axis=1, ascending=ascending, method="first")
    sel = rk <= n
    return sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def fund(field):
    """월말 PER/PBR/DIV 등 → 일봉 격자(월말 행에만 값).
    PER·PBR 은 DART 로 계산(2026-09-24, 출처들의 최근 분기 기준에 맞춤):
      PER = 월말 시총 ÷ 지배주주 순이익 TTM(최근 4분기 합), PBR = 월말 시총 ÷ 최근 분기말 지배주주 자본
      (지배주주 항목이 없는 별도재무 기업은 전체 순이익·자본). 적자·자본잠식은 0(= 해당 없음, pykrx 관례와 같게).
    DART 가 없는 구간(2016-04 이전)·종목만 pykrx(직전 사업연도 연간) 값으로 채운다. 나머지 항목은 pykrx."""
    f = K.load("fundamental")
    w = f.pivot_table(index="date", columns="code", values=field).reindex(columns=d.C.columns)
    if field not in ("PER", "PBR") or not (K.PANEL / "fin_ni_own_ttm.parquet").exists():
        return w
    load = lambda n: pd.read_parquet(K.PANEL / f"fin_{n}.parquet").reindex(index=w.index, columns=w.columns)
    den = (load("ni_own_ttm").fillna(load("ni_ttm")) if field == "PER"
           else load("equity_own").fillna(load("equity")))
    cap = d.CAP.reindex(index=w.index, columns=w.columns)
    pc = K.PANEL / "pref_cap.parquet"
    if pc.exists():   # 지배주주 순이익·자본은 우선주 몫도 포함 → 분자도 보통주+우선주 시총
        cap = cap + pd.read_parquet(pc).reindex(index=w.index, columns=w.columns).fillna(0.0)
    dart = (cap / den).where(den > 0, 0.0).where(den.notna() & cap.notna())
    return dart.combine_first(w)


def etf_px(*codes):
    e = K.load("etf")
    px = {k.lower(): pd.concat({c: e[c][k] for c in codes}, axis=1)
          for k in ("Open", "High", "Low", "Close")}
    return {k: v.where(v > 0) for k, v in px.items()}


# ═══════════════ 오버나잇 / 종가 ═══════════════
def _etf_overnight(code, extra=None):
    c = etf_px(code)["close"]
    e = c.notna()
    if extra is not None:
        e &= extra(c)
    return Spec(entry=e, entry_at="close", hold=1, exit_hold_at="open", slots=1, etf=True,
                meta={"px": (code,)})


@strat("ON01", "오버나잇", "코스닥150 ETF 종가매수→익일시가", "멘탈헷징 블로그(C1)", etf=True)
def on01(): return _etf_overnight("229200")


@strat("ON02", "오버나잇", "코스닥150 ETF 오버나잇 월·화·수만", "멘탈헷징 블로그(C1)", etf=True)
def on02(): return _etf_overnight("229200", lambda c: pd.DataFrame(c.index.dayofweek.isin([0, 1, 2])[:, None],
                                                                   index=c.index, columns=c.columns))


@strat("ON03", "오버나잇", "KODEX200 종가매수→익일시가", "C1 대형주 비교판", etf=True)
def on03(): return _etf_overnight("069500")


@strat("ON04", "오버나잇", "코스닥150 ETF 당일 -2% 하락 시 종가매수→익일시가", "systrader79 인터뷰(C4)", etf=True)
def on04(): return _etf_overnight("229200", lambda c: c.pct_change() <= -0.02)


@strat("ON05", "오버나잇", "상한가 마감 종가매수→익일시가", "최우석·한상일 2010(C7), 상따", since="2015-06-15",
       assume="상한가 잔량 체결 가능 가정(실제로는 체결 어려움)")
def on05(): return stock(d.R >= 0.295, entry_at="close", hold=1, exit_hold_at="open")


@strat("ON06", "오버나잇", "장대양봉 종가베팅→익일시가", "크몽 K1 전자책(C9) 근사",
       assume="장대양봉=등락+5%↑·종가 고가근접(상위 20%)·거래량 20일평균 2배↑·종가>MA20")
def on06():
    e = (d.R >= 0.05) & (close_pos(d.H, d.L, d.C) >= 0.8) & (d.V >= 2 * ma(d.V, 20)) & (d.C > ma(d.C, 20))
    return stock(e, entry_at="close", hold=1, exit_hold_at="open")


@strat("ON07", "오버나잇", "쌍끌이(외국인+기관 순매수) 종가베팅→익일시가", "space-cap/closing-price-bet(C10)",
       assume="당일 외국인·기관 동시 순매수 + 양봉 + 종가 레인지 상위 30%")
def on07():
    e = (d.FOR > 0) & (d.INST > 0) & (d.R > 0) & (close_pos(d.H, d.L, d.C) >= 0.7)
    return stock(e, entry_at="close", hold=1, exit_hold_at="open")


def _vb(k=0.5, ma_filter=False, n_top=50):
    rng = (d.H - d.L).shift(1)
    target = d.O + k * rng
    hit = d.H >= target
    if ma_filter:
        hit &= (target > ma(d.C, 5).shift(1)) & (target > ma(d.C, 10).shift(1))
    ep = np.maximum(d.O, target)
    return hit & top_trdval(n_top).shift(1, fill_value=False), ep


@strat("ON08", "오버나잇", "변동성돌파 k=0.5 → 익일시가 청산(거래대금 상위50)", "래리 윌리엄스, focalpoint94(C11)",
       assume="유니버스 = 전일 거래대금 상위 50")
def on08():
    e, ep = _vb()
    return Spec(entry=(e & d.U.shift(1, fill_value=False)).fillna(False), entry_at="intraday", entry_price=ep,
                hold=1, exit_hold_at="open")


@strat("ON09", "오버나잇", "변동성돌파 + 5·10일선 필터 → 익일시가", "INVESTAR 예제(B1) 필터 + C11")
def on09():
    e, ep = _vb(ma_filter=True)
    return Spec(entry=(e & d.U.shift(1, fill_value=False)).fillna(False), entry_at="intraday", entry_price=ep,
                hold=1, exit_hold_at="open")


# ═══════════════ 데이트레이딩 / 조건식 (일봉, 하루 1회) ═══════════════
@strat("DT01", "데이트레이딩", "레버리지 ETF 변동성돌파 당일청산(+5·10일선)", "INVESTAR 《파이썬 증권 데이터 분석》(B1)", etf=True)
def dt01():
    p = etf_px("122630", "233740")
    o, h, l, c = p["open"], p["high"], p["low"], p["close"]
    target = o + 0.5 * (h - l).shift(1)
    e = (h >= target) & (target > ma(c, 5).shift(1)) & (target > ma(c, 10).shift(1))
    return Spec(entry=e.fillna(False), entry_at="intraday", entry_price=np.maximum(o, target),
                hold=1, exit_hold_at="close", slots=2, etf=True, meta={"px": ("122630", "233740")})


@strat("DT02", "데이트레이딩", "변동성돌파 당일 종가청산(거래대금 상위50)", "B1 규칙의 개별종목판")
def dt02():
    e, ep = _vb()
    return Spec(entry=(e & d.U.shift(1, fill_value=False)).fillna(False), entry_at="intraday", entry_price=ep,
                hold=1, exit_hold_at="close")


@strat("CD01", "조건식", "세력매집 포착식", "brunch @nftmby(A2)", 
       assume="저점 반등=종가>20일 최저가×1.03 & 양봉. 청산은 공통(익절10/손절3/5일)")
def cd01():
    e = ((d.V >= 1.5 * ma(d.V, 5).shift(1)) & (d.FOR.rolling(3).min() > 0) & (d.INST.rolling(3).min() > 0)
         & (d.C > ll(d.L, 20) * 1.03) & (d.R > 0))
    return stock(e, **GENERIC_EXIT)


@strat("CD02", "조건식", "하승훈식 초기돌파", "tilnote 요약(A3)", assume="청산(원문 없음 → 단기 규칙 사전 고정): 신호일 저가 이탈 손절, 최대 3거래일")
def cd02():
    e = (d.C > ma(d.C, 240)) & (d.V >= 5 * d.V.shift(1)) & (d.R >= 0.05) & (d.C >= hh(d.H, 60))
    return stock(e, **SHORT_EXIT())


@strat("CD03", "조건식", "창원개미 4% 조건식(코스닥)", "tilnote 요약(A4)", assume="청산(원문 없음 → 단기 규칙 사전 고정): 신호일 저가 이탈 손절, 최대 3거래일")
def cd03():
    kq = pd.Series(d.C.columns.map(d.META["market"]).str.contains("KOSDAQ"), index=d.C.columns)
    raw_close = K.load("close")  # 주가 조건은 실제(미수정) 가격 기준
    e = (d.H >= 0.95 * hh(d.H, 15)) & (d.V >= 200_000) & (raw_close >= 800) & (raw_close <= 150_000)
    e &= pd.DataFrame(np.broadcast_to(kq.to_numpy(), e.shape), index=e.index, columns=e.columns)
    return stock(e, **SHORT_EXIT())


@strat("CD04", "조건식", "시가갭 +1~10% 시가매수(동시호가식 근사)", "Curookie gist(A6)",
       assume="09:00 시가로 진입 판단·체결, 손절 1%·익절 3%·당일 종가청산")
def cd04():
    gap = d.O / d.C.shift(1) - 1
    e = between(gap, 0.01, 0.10) & (d.V.shift(1) >= 50_000) & d.U.shift(1, fill_value=False)
    return Spec(entry=e.fillna(False), entry_at="intraday", entry_price=d.O, sl=0.01, tp=0.03,
                hold=1, exit_hold_at="close")


@strat("CD05", "조건식", "20/60 골든크로스 당일 전환", "bangulnote(A8)", assume="청산(원문 없음 → 단기 규칙 사전 고정): 신호일 저가 이탈 손절, 최대 3거래일")
def cd05(): return stock(crossup(ma(d.C, 20), ma(d.C, 60)), **SHORT_EXIT())


# ═══════════════ 스윙 ═══════════════
@strat("SW01", "스윙", "터틀 55일 돌파 / 20일 이탈 청산, 2ATR 손절", "터틀 System2, kwjy_turtle_trading")
def sw01():
    a = atr(d.H, d.L, d.C, 20)
    return stock(d.C > hh(d.H, 55).shift(1), exit=d.C < ll(d.L, 20).shift(1), sl=(2 * a / d.C))


@strat("SW02", "스윙", "돈치안 20일 돌파 + 2ATR 손절, 최대 7일", "luckyD333/kospi-swing-scanner S3")
def sw02():
    a = atr(d.H, d.L, d.C, 20)
    return stock(d.C > hh(d.H, 20).shift(1), sl=(2 * a / d.C), hold=7)


@strat("SW03", "스윙", "종가 20일선 상향 돌파 매수 / 하향 이탈 매도", "systrader79(168), 2003 KCI 이평 논문")
def sw03(): return stock(crossup(d.C, ma(d.C, 20)), exit=d.C < ma(d.C, 20))


@strat("SW04", "스윙", "골든크로스 20/60, 5일 보유", "책 요약 블로그(ksa30702)")
def sw04(): return stock(crossup(ma(d.C, 20), ma(d.C, 60)), hold=5)


@strat("SW05", "스윙", "골든크로스 20/60, 10일 보유", "책 요약 블로그(ksa30702)")
def sw05(): return stock(crossup(ma(d.C, 20), ma(d.C, 60)), hold=10)


def _rs():
    r = lambda n: d.C / d.C.shift(n) - 1
    w = 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)
    return w.where(d.U).rank(axis=1, pct=True) * 100


@strat("SW06", "스윙", "미너비니 트렌드템플릿+VCP 돌파(손절7%, 10일)", "younghwan91/swing-it minervini_prop")
def sw06():
    m50, m150, m200 = ma(d.C, 50), ma(d.C, 150), ma(d.C, 200)
    e = ((d.C > m50) & (m50 > m150) & (m150 > m200) & (m200 > m200.shift(20))
         & (d.C >= 0.9 * hh(d.C, 252)) & (d.C >= 1.25 * ll(d.C, 252)) & (_rs() >= 70)
         & (d.C > hh(d.H, 50).shift(1)) & (ma(d.V, 5).shift(1) < 0.7 * ma(d.V, 50).shift(1)))
    return stock(e, sl=0.07, hold=10)


@strat("SW07", "스윙", "52주 신고가 돌파(손절7%, 익절21%)", "깡토 『손실은 짧게 수익은 길게』",
       assume="손절 5~9% 중 7%, 손익비 1:3 → 익절 21%, 최대 60일")
def sw07(): return stock(d.C > hh(d.H, 250).shift(1), sl=0.07, tp=0.21, hold=60)


@strat("SW08", "스윙", "RSI(2) 역추세 Connors형", "『쉽게 따라 만드는 파이썬 주식 자동매매 시스템』 4장",
       assume="15시 체결을 종가 진입으로 근사, '현재가>매수가' 조건은 생략(RSI2>80 종가 청산)")
def sw08():
    e = (ma(d.C, 20) > ma(d.C, 60)) & (rsi(d.C, 2) < 5) & (d.C / d.C.shift(2) - 1 < -0.02)
    return stock(e, entry_at="close", exit=rsi(d.C, 2) > 80, exit_at="close")


@strat("SW09", "스윙", "정배열 눌림목(RSI14<30 반등)", "swing-it pullback_prop", assume="RSI 는 원문대로 SMA 방식")
def sw09():
    m50, m150, m200 = ma(d.C, 50), ma(d.C, 150), ma(d.C, 200)
    e = ((d.C > m50) & (m50 > m150) & (m150 > m200) & (m200 > m200.shift(20))
         & (rsi_sma(d.C, 14).shift(1) < 30) & (d.C > d.C.shift(1)))
    return stock(e, sl=0.04, tp=0.05, hold=8)


@strat("SW10", "스윙", "IBS<0.05 후 익일 -1.5% 지정가 매수", "systrader79(229)",
       assume="원출처의 가치점수 필터는 생략(시총 1조 미만만 적용). 청산(원문): 종가가 매수가 +1% 이상 또는 종가>전일고가면 익일 시가, 최대 5일")
def sw10():
    ibs = close_pos(d.H, d.L, d.C)
    setup = (ibs < 0.05) & (d.CAP < 1e12) & d.U
    limit = d.C * 0.985
    e = (setup.shift(1, fill_value=False) & (d.L <= limit.shift(1))).fillna(False)
    ep = np.minimum(d.O, limit.shift(1))
    last_ep = ep.where(e).ffill()                     # 보유 중 종목의 매수가(종목당 포지션 1개)
    return Spec(entry=e, entry_at="intraday", entry_price=ep,
                exit=(d.C >= last_ep * 1.01) | (d.C > d.H.shift(1)), hold=5)


@strat("SW11", "스윙", "볼린저 하단 이탈 후 복귀 매수 / 상단 매도", "책 요약(ksa30702), 성과검증센터")
def sw11():
    m, s = ma(d.C, 20), d.C.rolling(20).std()
    lo, up = m - 2 * s, m + 2 * s
    return stock((d.C.shift(1) < lo.shift(1)) & (d.C > lo), exit=d.C > up, hold=20)


@strat("SW12", "스윙", "이격도 95 이하 과매도 반등", "systrader79(443·444)",
       assume="이격도(C/MA20)≤95 & 양봉 진입, MA20 회복 시 청산, 최대 10일(가정). 원저자(stock79 444)는 이 아이디어를 성과 불량으로 기각")
def sw12(): return stock((d.C / ma(d.C, 20) <= 0.95) & (d.R > 0), exit=d.C >= ma(d.C, 20), hold=10)


@strat("SW13", "스윙", "거래량 급증 돌파(20일평균 2배, 양봉, MA20 위)", "성과검증센터 거래량편, B6",
       assume="손절 5%, 최대 10일")
def sw13(): return stock((d.V >= 2 * ma(d.V, 20).shift(1)) & (d.R > 0) & (d.C > ma(d.C, 20)), sl=0.05, hold=10)


@strat("SW14", "스윙", "외국인·기관 3일 연속 동반 순매수", "유진·장순재 2012, 성과검증센터", 
       assume="손절 5%, 최대 5일")
def sw14(): return stock((d.FOR.rolling(3).min() > 0) & (d.INST.rolling(3).min() > 0), sl=0.05, hold=5)


# ═══════════════ HollyKR 자체 전략 (종가 진입, 카테고리별 ATR 목표/손절) ═══════════════
def _holly(e, tgt, stp, hold=20):
    a = atr(d.H, d.L, d.C, 14)
    return stock(e, entry_at="close", tp=tgt * a / d.C, sl=stp * a / d.C, hold=hold)


HOLLY_ASSUME = "HollyKR 신호 로직 이식. 부분익절·트레일링은 생략, 최대 20일"
HOLLY_FIXED = HOLLY_ASSUME + ". 목표·손절은 원문대로 고정 %"


def _holly_fixed(e, tp, sl, hold=20):
    return stock(e, entry_at="close", tp=tp, sl=sl, hold=hold)


def _stage2():
    """HollyKR _check_stage_2: 종가>MA200, 종가>MA50, MA50>MA200, MA200 1개월 전보다 상승."""
    m50, m200 = ma(d.C, 50), ma(d.C, 200)
    return (d.C > m200) & (d.C > m50) & (m50 > m200) & (m200 > m200.shift(21))


@strat("HK01", "HollyKR", "close_to_a_cross(MA5×MA50 골든크로스+2%)", "HollyKR", assume=HOLLY_ASSUME)
def hk01():
    vr = d.V / ma(d.V, 50)
    e = crossup(ma(d.C, 5), ma(d.C, 50)) & (d.R >= 0.02) & _stage2() & between(vr, 1.3, 6.0)
    return _holly(e, 5.0, 2.0)


@strat("HK02", "HollyKR", "weinstein_stage(1→2단계 전환)", "HollyKR", assume=HOLLY_ASSUME)
def hk02():
    m150 = ma(d.C, 150)
    slope = m150 - m150.shift(22)
    was1 = ((slope.shift(5).abs() / m150.shift(5) < 0.01) & ((d.C.shift(5) - m150.shift(5)).abs() / m150.shift(5) < 0.05))
    e = (d.C > m150) & (slope > 0) & was1 & (d.V > 2 * ma(d.V, 20)) & (ll(d.C, 5) > ll(m150, 5))
    return _holly(e, 5.0, 2.0)


@strat("HK03", "HollyKR", "tailwind(정배열 MA20 눌림 양봉)", "HollyKR", assume=HOLLY_ASSUME)
def hk03():
    m5, m20, m60 = ma(d.C, 5), ma(d.C, 20), ma(d.C, 60)
    e = (_stage2() & (m5 > m20) & (m20 > m60) & (m5 > m5.shift(5)) & (m20 > m20.shift(5)) & (m60 > m60.shift(5))
         & (d.C >= m20) & (d.C / m20 - 1 <= 0.02) & (d.C > d.O) & between(d.V / ma(d.V, 50), 1.3, 5.0))
    return _holly(e, 5.0, 2.0)


@strat("HK04", "HollyKR", "wake_up_call(20일 신고가·MA5>MA20·양봉)", "HollyKR", assume=HOLLY_ASSUME)
def hk04():
    e = ((d.C > ma(d.C, 200)) & (d.C >= 0.99 * hh(d.H, 20)) & (ma(d.C, 5) > ma(d.C, 20))
         & between(d.V / ma(d.V, 50), 1.5, 5.0) & (d.C > d.O))
    return _holly(e, 5.0, 2.0)


@strat("HK05", "HollyKR", "darvas_box 근사(20일 박스 15% 이내 돌파)", "HollyKR",
       assume=HOLLY_ASSUME + ". 박스 탐지는 20일 고저 폭 15% 이내로 단순화")
def hk05():
    top, bot = hh(d.H, 20).shift(1), ll(d.L, 20).shift(1)
    box = (top - bot) / ((top + bot) / 2) <= 0.15
    e = box & (d.C > top * 1.005) & (d.V > 2 * ma(d.V, 20)) & (close_pos(d.H, d.L, d.C) >= 0.7)
    return _holly(e, 5.0, 2.0)


@strat("HK06", "HollyKR", "volume_doesnt_lie(갭업5%+거래량 2.5~6배)", "HollyKR", assume=HOLLY_ASSUME)
def hk06():
    gap = d.O / d.C.shift(1) - 1
    vr = d.V / ma(d.V, 50)
    e = ((d.C > ma(d.C, 200)) & (gap >= 0.05) & between(vr, 2.5, 6.0) & (d.C >= d.O)
         & (close_pos(d.H, d.L, d.C) >= 0.6))
    return _holly(e, 4.0, 1.6)


@strat("HK07", "HollyKR", "staggering_volume(거래량 3배+20일 신고가)", "HollyKR", assume=HOLLY_FIXED + "(+7%/-5%)")
def hk07():
    e = (d.V >= 3 * ma(d.V, 20)) & (d.C > hh(d.H, 20).shift(1)) & (close_pos(d.H, d.L, d.C) >= 0.7)
    return _holly_fixed(e, 0.07, 0.05)


@strat("HK08", "HollyKR", "quarterback(20% 풀백 후 RSI 전환)", "HollyKR", assume=HOLLY_ASSUME)
def hk08():
    r14, r5 = rsi(d.C, 14), rsi(d.C, 5)
    e = (d.C <= 0.8 * hh(d.H, 20)) & between(r14, 30, 50) & (r5 > r14) & (d.C > d.O)
    return _holly(e, 3.0, 1.5)


@strat("HK09", "HollyKR", "trend_play(6중 정배열, MA5 근접)", "HollyKR",
       assume=HOLLY_FIXED + "(+7%, 손절 = max(MA20, 진입가×0.95))")
def hk09():
    ms = [ma(d.C, n) for n in (5, 10, 20, 50, 100, 200)]
    e = d.C.notna()
    for a, b in zip(ms, ms[1:]):
        e &= a > b
    for m in ms:
        e &= m > m.shift(5)
    e &= (ms[3] > ms[3].shift(10)) & ((d.C / ms[0] - 1).abs() <= 0.02)
    return _holly_fixed(e, 0.07, (1 - ms[2] / d.C).clip(0.001, 0.05))


@strat("HK10", "HollyKR", "nice_chart(전일고가 돌파·RSI 45~65·거래량 1.5배)", "HollyKR", assume=HOLLY_FIXED + "(+6%/-4%)")
def hk10():
    r = rsi(d.C, 14)
    e = (d.C > d.H.shift(1)) & (ma(d.C, 5) > ma(d.C, 20)) & (r > 45) & (r < 65) & (d.V > 1.5 * ma(d.V, 20))
    return _holly_fixed(e, 0.06, 0.04)


# ═══════════════ 중장기 (월말 신호 → 익일 시가 리밸런싱) ═══════════════
def RB(weights, etf=False, px=None):
    return dict(kind="rebalance", weights=weights, etf=etf, px=px)


def _me(): return month_ends(d.C.index)
def _july(): return pd.DatetimeIndex([x for x in _me() if x.month == 6])   # 6월말 신호 → 7월 초 체결


def _small(dates):
    cap = d.CAP.where(d.U).reindex(dates)
    return cap.le(cap.quantile(0.2, axis=1), axis=0)


@strat("LT01", "중장기", "소형주(유니버스 하위20%) + 저PBR 25종목, 연1회", "강환국 『할 수 있다! 퀀트 투자』",
       assume="소형주 = 시총 1,000억 이상 유니버스 안 하위 20%(원출처는 전 종목 하위 20%)")
def lt01():
    dt = _july()
    pbr = fund("PBR").reindex(dt)
    return RB(pick(pbr.where(pbr > 0.2), _small(dt), 25, dt, ascending=True))


@strat("LT02", "중장기", "밸류 합성(PER·PBR·배당 순위합) 소형주 30종목, 연1회", "강환국 슈퍼가치 근사",
       assume="PSR·PCR 은 재무데이터 부재로 제외, 배당수익률로 대체")
def lt02():
    dt = _july()
    per, pbr, div = (fund(f).reindex(dt) for f in ("PER", "PBR", "DIV"))
    score = (per.where(per > 0).rank(axis=1) + pbr.where(pbr > 0).rank(axis=1)
             + div.rank(axis=1, ascending=False))
    return RB(pick(score, _small(dt), 30, dt, ascending=True))


@strat("LT03", "중장기", "저PER+저PBR 20종목, 월간", "아이투자 드레먼식")
def lt03():
    dt = _me()
    per, pbr = fund("PER").reindex(dt), fund("PBR").reindex(dt)
    score = per.where(per > 0).rank(axis=1) + pbr.where(pbr > 0).rank(axis=1)
    return RB(pick(score, d.U, 20, dt, ascending=True))


@strat("LT04", "중장기", "저변동성 30종목(60일), 월간", "고봉찬·김진우 저변동성 논문")
def lt04():
    dt = _me()
    vol = d.R.rolling(60, min_periods=40).std()
    return RB(pick(vol, d.U, 30, dt, ascending=True))


def _mom(n=252, skip=21): return d.C.shift(skip) / d.C.shift(n) - 1


@strat("LT05", "중장기", "12-1 모멘텀 상위 30, 월간", "henryquant, 이창준·김창하 2018")
def lt05():
    dt = _me()
    return RB(pick(_mom(), d.U, 30, dt))


@strat("LT06", "중장기", "K-ratio 모멘텀 상위 30, 월간", "henryquant K-ratio")
def lt06():
    dt = _me()
    y = np.log(d.C)
    n = 252
    x = np.arange(n) - (n - 1) / 2
    rows = {}
    for t in dt:
        i = y.index.get_loc(t)
        if i < n:
            continue
        Y = y.iloc[i - n + 1:i + 1]
        ok = Y.notna().all()
        Yc = Y - Y.mean()
        slope = (Yc.mul(x, axis=0)).sum() / (x ** 2).sum()
        resid = Yc - np.outer(x, slope)
        se = np.sqrt((resid ** 2).sum() / (n - 2) / (x ** 2).sum())
        rows[t] = (slope / se).where(ok)
    kr = pd.DataFrame(rows).T.reindex(dt)
    return RB(pick(kr, d.U, 30, dt))


@strat("LT07", "중장기", "개별종목 듀얼모멘텀(12-1>0 & 상위 30), 월간", "henryquant dual momentum")
def lt07():
    dt = _me()
    m = _mom()
    return RB(pick(m.where(m > 0), d.U, 30, dt))


@strat("LT08", "중장기", "밸류+모멘텀(저PBR 하위 20% 중 12-1 상위 30), 월간", "박지윤 2018")
def lt08():
    dt = _me()
    pbr = fund("PBR").reindex(dt)
    cheap = pbr.where(pbr > 0).le(pbr.where(pbr > 0).quantile(0.2, axis=1), axis=0)
    return RB(pick(_mom(), cheap & d.U.reindex(dt), 30, dt))


@strat("LT09", "중장기", "고배당 30종목, 월간", "윤철수 2018, 유제완·안성진 2025")
def lt09():
    dt = _me()
    return RB(pick(fund("DIV").reindex(dt), d.U, 30, dt))


@strat("LT10", "중장기", "소형주 효과(유니버스 하위 20% 동일비중), 월간", "엄철준 외 2024",
       assume="유니버스(1,000억 이상) 안 하위 20%")
def lt10():
    dt = _me()
    s = _small(dt)
    return RB(s.div(s.sum(axis=1), axis=0).fillna(0.0))


@strat("LT11", "중장기", "1월 소형주(12월말 매수, 1월말 매도)", "엄철준·이우백·박종원 2014")
def lt11():
    dt = _me()
    s = _small(dt)
    w = s.div(s.sum(axis=1), axis=0).fillna(0.0)
    w.loc[w.index.month != 12] = 0.0
    return RB(w)


@strat("LT12", "중장기", "1개월 단기반전(하위 30), 월간", "엄철준·박종원 2023")
def lt12():
    dt = _me()
    return RB(pick(d.C / d.C.shift(21) - 1, d.U, 30, dt, ascending=True))


@strat("LT13", "중장기", "52주 고가 근접도 상위 30, 월간", "George-Hwang, 정대성·류호영 2021")
def lt13():
    dt = _me()
    return RB(pick(d.C / hh(d.C, 250), d.U, 30, dt))


@strat("LT14", "중장기", "기관 순매수 가속도 상위 10%, 월간", "swing-it inst_flow_accel", 
       assume="(최근20일-직전20일 기관순매수)/시총, 상위 30종목")
def lt14():
    dt = _me()
    s20 = d.INST.rolling(20, min_periods=15).sum()
    acc = (s20 - s20.shift(20)) / (d.CAP / 1e6)
    return RB(pick(acc, d.U, 30, dt))


# ═══════════════ ETF 자산배분 ═══════════════
def _etf_w(codes, fn, freq="M"):
    p = etf_px(*codes)
    c = p["close"]
    idx = c.dropna(how="any").index
    dts = month_ends(idx) if freq == "M" else pd.DatetimeIndex([x for x in month_ends(idx) if x.month in (3, 6, 9, 12)])
    w = pd.DataFrame({t: fn(t, c) for t in dts}).T.reindex(columns=list(codes)).fillna(0.0)
    return RB(w, etf=True, px=p)


@strat("AA01", "자산배분", "할로윈(11~4월 KODEX200, 5~10월 현금)", "skyatlastreasure 백테스트", etf=True)
def aa01():
    return _etf_w(["069500"], lambda t, c: {"069500": 1.0 if t.month in (10, 11, 12, 1, 2, 3) else 0.0})


AW = {"143850": .175, "069500": .175, "305080": .25, "148070": .25, "132030": .15}


@strat("AA02", "자산배분", "K-올웨더(미주17.5·한주17.5·미채25·한채25·금15), 분기", "김성일 『마법의 연금 굴리기』", etf=True,
       assume="미국주식=TIGER 미국S&P500선물(H), 미국채=TIGER 미국채10년선물")
def aa02(): return _etf_w(list(AW), lambda t, c: AW, freq="Q")


@strat("AA03", "자산배분", "K-올웨더+할로윈(위험자산 5~10월 20%, 11~4월 50%)", "luappa 블로그", etf=True)
def aa03():
    def fn(t, c):
        risk = 0.5 if t.month in (10, 11, 12, 1, 2, 3) else 0.2
        return {"143850": risk / 2, "069500": risk / 2,
                "305080": (1 - risk) * 0.25 / 0.65, "148070": (1 - risk) * 0.25 / 0.65,
                "132030": (1 - risk) * 0.15 / 0.65}
    return _etf_w(list(AW), fn)


@strat("AA04", "자산배분", "주식·채권 변동성 역가중(KODEX200·국고채10년), 월간", "systrader79(intelliquant 490)", etf=True)
def aa04():
    def fn(t, c):
        v = c.loc[:t].pct_change().tail(60).std()
        iv = 1 / v
        return (iv / iv.sum()).to_dict()
    return _etf_w(["069500", "148070"], fn)


@strat("AA05", "자산배분", "영구 포트폴리오(주식·채권·금·달러 각 25%), 분기", "해리 브라운, 헤럴드경제", etf=True,
       assume="달러 = KIWOOM 미국달러선물(138230, 2011년 상장)")
def aa05(): return _etf_w(["069500", "148070", "132030", "138230"], lambda t, c: dict.fromkeys(c.columns, .25), freq="Q")


@strat("AA06", "자산배분", "ETF 듀얼모멘텀(KODEX200 vs 미국S&P, 절대모멘텀 실패 시 국고채)", "안토나치 한국 ETF판", etf=True)
def aa06():
    def fn(t, c):
        r = c.loc[:t].iloc[-1] / c.loc[:t].iloc[-253] - 1 if len(c.loc[:t]) > 253 else None
        if r is None:
            return {}
        if r[["069500", "143850", "153130"]].isna().any():
            return {}
        best = r[["069500", "143850"]].idxmax()
        return {best: 1.0} if r[best] > r["153130"] else {"148070": 1.0}
    return _etf_w(["069500", "143850", "148070", "153130"], fn)


# ═══════════════ 2차 조사: 중장기 팩터 (2026-09-23 추가) ═══════════════
def _qe(): return pd.DatetimeIndex([x for x in _me() if x.month in (3, 6, 9, 12)])


def _roe():
    per, pbr = fund("PER"), fund("PBR")
    return (pbr / per).where(per > 0)          # ROE ≈ PBR/PER (적자는 NaN)


def _turnover(n=21): return (d.TV / d.CAP).rolling(n, min_periods=15).mean()
def _max(n=21): return d.R.rolling(n, min_periods=15).max()


def _rk(x, asc=True): return x.rank(axis=1, ascending=asc, pct=True)


def mixed_px(*codes):
    """주식 수정주가 + ETF 가격을 한 가격표로(시장필터·할로윈 결합 전략용)."""
    e = etf_px(*codes)
    return {k: pd.concat([d.__getattr__({"open": "O", "high": "H", "low": "L", "close": "C"}[k]),
                          e[k].reindex(d.C.index)], axis=1) for k in ("open", "high", "low", "close")}


def _market_ok(dates):
    """KODEX200 월말 종가 > 10개월 이동평균 → True."""
    c = etf_px("069500")["close"]["069500"]
    me = c.groupby(c.index.to_period("M")).last()
    ok = (me > me.rolling(10).mean())
    return pd.Series(ok.reindex(dates.to_period("M")).to_numpy(), index=dates).fillna(False)


def _with_filter(w, dates, bond="148070"):
    """시장 필터: KOSPI 약세 달엔 주식 대신 국고채10년 100%."""
    ok = _market_ok(dates)
    w = w.reindex(dates).fillna(0.0)
    w.loc[~ok] = 0.0
    w[bond] = np.where(ok, 0.0, 1.0)
    return RB(w, px=mixed_px(bond))


def _with_halloween(w, dates, bond="148070"):
    w = w.reindex(dates).fillna(0.0)
    summer = np.isin(dates.month, [4, 5, 6, 7, 8, 9])      # 4월말 신호 → 5월초~10월말 채권
    w.loc[summer] = 0.0
    w[bond] = np.where(summer, 1.0, 0.0)
    return RB(w, px=mixed_px(bond))


def _lowpe_pb(dt):
    per, pbr = fund("PER").reindex(dt), fund("PBR").reindex(dt)
    return per.where(per > 0).rank(axis=1) + pbr.where(pbr > 0).rank(axis=1)


@strat("LT15", "중장기", "고ROE(PBR/PER)+저PER 30종목, 분기", "하나증권 실전퀀트 2024.2(고ROE 1위), 밸류업")
def lt15():
    dt = _qe()
    per = fund("PER").reindex(dt)
    score = _rk(_roe().reindex(dt), asc=False) + _rk(per.where(per > 0))
    return RB(pick(score, d.U, 30, dt, ascending=True))


@strat("LT16", "중장기", "한국판 JPX프라임150(시총상위500 중 ROE 상위75 + PBR≥1 시총상위75), 분기",
       "하나증권 실전퀀트 2024.2.27", assume="COE 8% 상수 가정 → ROE 상위와 동일 순위")
def lt16():
    dt = _qe()
    cap = d.CAP.reindex(dt).where(d.U.reindex(dt))
    top500 = cap.rank(axis=1, ascending=False) <= 500
    roe = _roe().reindex(dt).where(top500)
    a = roe.rank(axis=1, ascending=False) <= 75
    pbr = fund("PBR").reindex(dt)
    b = cap.where(top500 & (pbr >= 1)).rank(axis=1, ascending=False) <= 75
    sel = a | b
    return RB(sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0))


@strat("LT17", "중장기", "밸류업 복합(저PER·저PBR·고ROE·고배당·배당성향) 30종목, 분기",
       "하나증권 코리아밸류업150 시뮬레이션", assume="현금흐름 지표는 DART 부재로 제외")
def lt17():
    dt = _qe()
    per, pbr, div, dps, eps = (fund(f).reindex(dt) for f in ("PER", "PBR", "DIV", "DPS", "EPS"))
    payout = (dps / eps).where(eps > 0)
    score = (_rk(per.where(per > 0)) + _rk(pbr.where(pbr > 0)) + _rk(_roe().reindex(dt), False)
             + _rk(div, False) + _rk(payout, False))
    return RB(pick(score, d.U, 30, dt, ascending=True))


@strat("LT18", "중장기", "저변동성+저회전율+저MAX 복합 30종목, 월간", "강장구·심명화(MAX), JDQS 2020 한국 이상현상")
def lt18():
    dt = _me()
    vol = d.R.rolling(60, min_periods=40).std()
    score = _rk(vol.reindex(dt)) + _rk(_turnover().reindex(dt)) + _rk(_max().reindex(dt))
    return RB(pick(score, d.U, 30, dt, ascending=True))


@strat("LT19", "중장기", "저PBR+저MAX+저회전율 30종목, 월간", "JDQS 2020(가치·거래마찰 재현율 최고)")
def lt19():
    dt = _me()
    pbr = fund("PBR").reindex(dt)
    score = _rk(pbr.where(pbr > 0)) + _rk(_turnover().reindex(dt)) + _rk(_max().reindex(dt))
    return RB(pick(score, d.U, 30, dt, ascending=True))


@strat("LT20", "중장기", "EPS증가/주가(PIR 근사)+저PER 30종목, 분기", "인텔리퀀트 #981 PIR",
       assume="영업이익 대신 EPS 변화(12개월 전 스냅샷 대비)/주가")
def lt20():
    dt = _qe()
    eps = fund("EPS")
    deps = (eps - eps.shift(12)).reindex(dt)
    price = K.load("close").reindex(dt)
    per = fund("PER").reindex(dt)
    score = _rk(deps / price, False) + _rk(per.where(per > 0))
    return RB(pick(score, d.U, 30, dt, ascending=True))


@strat("LT21", "중장기", "신F-스코어(흑자·영업현금흐름+·무증자)+저PBR 25종목, 연1회", "강환국/인텔리퀀트 #583",
       assume="흑자·영업CF 는 DART TTM(2016-05 전은 pykrx EPS>0, 영업CF 조건 없음). "
              "무증자 = 수정주가 기준 주식수(시총÷수정종가) 1년 증가 1% 이내 → 액면분할·무상증자는 제외, 유상증자·CB 전환만 잡힘")
def lt21():
    dt = _july()
    shares = d.CAP / d.C                              # 수정주가 기준: 분할·무상증자에 불변
    no_issue = (shares / shares.shift(250) <= 1.01).reindex(dt)
    ni = pd.read_parquet(K.PANEL / "fin_ni_own_ttm.parquet").reindex(index=dt, columns=d.C.columns)
    ocf = pd.read_parquet(K.PANEL / "fin_ocf_ttm.parquet").reindex(index=dt, columns=d.C.columns)
    profit = (ni > 0).where(ni.notna(), fund("EPS").reindex(dt) > 0)
    cash_ok = (ocf > 0).where(ocf.notna(), True)
    pbr = fund("PBR").reindex(dt)
    ok = profit.astype(bool) & cash_ok.astype(bool) & no_issue & d.U.reindex(dt)
    return RB(pick(pbr.where(pbr > 0), ok, 25, dt, ascending=True))


@strat("LT22", "중장기", "SUE(표준화 EPS 서프라이즈) 상위 30, 월간", "정찬식·김순호 2015 PEAD",
       assume="SUE=(EPS-12개월전)/과거 24개월 변화 표준편차. EPS 가 연 1회(5월) 갱신이라 SUE 는 갱신 시 계산해 12개월 유지")
def lt22():
    dt = _me()
    eps = fund("EPS")
    dy = eps - eps.shift(12)
    sd = dy.rolling(24, min_periods=12).std()
    sue = (dy / sd.replace(0, np.nan)).where(eps.ne(eps.shift(1))).ffill(limit=12)
    return RB(pick(sue.reindex(dt), d.U, 30, dt))


@strat("LT23", "중장기", "Amihud 비유동성 상위 30(유니버스 내), 월간", "강장구·정기호 2018 한국증권학회지")
def lt23():
    dt = _me()
    ami = (d.R.abs() / d.TV.replace(0, np.nan)).rolling(21, min_periods=15).mean()
    return RB(pick(ami.reindex(dt), d.U, 30, dt))


@strat("LT24", "중장기", "저회전율 30종목, 월간", "JDQS 2020 한국 이상현상(1개월 회전율)")
def lt24():
    dt = _me()
    return RB(pick(_turnover().reindex(dt), d.U, 30, dt, ascending=True))


@strat("LT25", "중장기", "저변동성 30 + KOSPI 10개월선 필터(약세 시 국고채)", "인텔리퀀트 #869 시장필터 + 페이버",
       assume="필터는 KODEX200 월말 종가 vs 10개월 이동평균")
def lt25():
    dt = _me()
    vol = d.R.rolling(60, min_periods=40).std()
    return _with_filter(pick(vol, d.U, 30, dt, ascending=True), dt)


@strat("LT26", "중장기", "저PER+저PBR 20 + KOSPI 10개월선 필터", "인텔리퀀트 #869 시장필터")
def lt26():
    dt = _me()
    return _with_filter(pick(_lowpe_pb(dt), d.U, 20, dt, ascending=True), dt)


@strat("LT27", "중장기", "저PER+저PBR 20 + 할로윈(5~10월 국고채)", "인텔리퀀트 #869 할로윈 결합")
def lt27():
    dt = _me()
    return _with_halloween(pick(_lowpe_pb(dt), d.U, 20, dt, ascending=True), dt)


@strat("LT28", "중장기", "저변동성 30 + 할로윈(5~10월 국고채)", "인텔리퀀트 #869 할로윈 결합")
def lt28():
    dt = _me()
    vol = d.R.rolling(60, min_periods=40).std()
    return _with_halloween(pick(vol, d.U, 30, dt, ascending=True), dt)


import strategies2  # noqa: E402,F401  2차 조사 전략 등록
if (K.PANEL / "fin_gp_ttm.parquet").exists():
    import strategies3  # noqa: E402,F401  DART 재무 전략(재무 패널이 있을 때만)
    import strategies4  # noqa: E402,F401  3차 조사 전략(방어·보유·이상현상·감마)
if (K.PANEL.parent / "dart" / "disclosures.parquet").exists():
    import strategies5  # noqa

AUDIT_NOTES = {  # 2026-09-24 전수 감사에서 찾은 설명 누락(동작은 그대로)
    "AA02": "비중은 책의 5자산판. 원저 다른 판(채권17.5·현금15 6자산)과 다를 수 있음(원문 표 미대조)",
    "AA03": "위험자산 몫은 미국·한국 주식 1:1",
    "AA14": "미국채10년선물(305080)이 공격·방어 목록에 모두 있음(원문 구조 그대로)",
    "DF03": "위험균등 비중은 5자산 전체로 정규화 후 추세 이탈 자산 몫만 현금(AllocateSmartly 규칙과 일치 확인)",
    "LT01": "PBR 0.2 이하는 제외(부실 전조 방어)",
    "LT02": "소형 = 유니버스(시총 1,000억 이상) 안 하위 20%. 원출처는 전종목 하위 20%",
    "LT11": "소형 = 유니버스(시총 1,000억 이상) 안 하위 20%. 원출처는 전종목 하위 20%",
    "LT13": "52주 고가 근접도는 종가 기준",
    "LT14": "상위 10% 대신 고정 30종목",
    "LT16": "ROE 상위 75와 PBR≥1 대형 상위 75의 합집합이라 중복 제외 후 약 130종목",
    "LT35": "두 6개월 모멘텀 구간이 일부 겹침(원문 확인 불가)",
    "LT42": "무증자 = 수정주가 기준 주식수(시총/수정종가) 1년 증가율 1% 이하(분할·무상증자 제외)",
    "SW04": "손절 없이 보유기간만(원출처 확인 불가)",
    "SW05": "손절 없이 보유기간만(원출처 확인 불가)",
    "CD03": "이름의 4% 는 원문 제목 그대로(조건은 15일 신고가 5% 이내 근접)",
    "ON05": "엔진의 상한가 종가매수 차단에 걸려 신호의 약 96% 는 체결되지 않음(체결 비현실로 순위 제외)",
    "EV02": "정정공시 제외, 슬롯 10", "EV03": "정정공시 제외, 슬롯 10",
    "EV04": "정정공시 제외, 슬롯 10. 감사의견 공시는 의견 종류 구분 없이 악재로 봄",
    "EV05": "정정공시 제외, 슬롯 10. 감사의견 공시는 의견 종류 구분 없이 악재로 봄",
    "EV06": "같은 날 연결·별도 잠정실적이 둘 다 오면 하나라도 조건 충족 시 신호",
    "EV07": "같은 날 연결·별도 잠정실적이 둘 다 오면 하나라도 조건 충족 시 신호",
    "EV08": "같은 날 연결·별도 잠정실적이 둘 다 오면 하나라도 조건 충족 시 신호. 감사의견 공시는 종류 구분 없이 악재",
}
for _sid, _note in AUDIT_NOTES.items():
    if _sid in REG:
        REG[_sid]["assume"] = "; ".join(x for x in (REG[_sid]["assume"], _note) if x)
