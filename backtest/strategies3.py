"""DART 재무 기반 전략. data/panel/fin_*.parquet(dart_panel.py 산출)이 있을 때만 strategies.py 가 import 한다.

재무값은 월말 시점에 '법정 제출기한이 지나 공개된' 가장 최근 보고서만 쓴다(dart_panel 참고).
DART 일괄 데이터가 2015 사업연도부터라 이 전략들은 2016년 4월 이후부터 신호가 나온다.
"""
import numpy as np
import pandas as pd

import kr_data as K
from strategies import RB, _me, _qe, d, fund, mixed_px, pick, strat
from strategies import _july  # noqa: F401  (연 1회 = 6월말 신호)
from strategies import _with_halloween

SINCE = "2016-05-01"
# DART 일괄 원천에 2015 사업연도 1~3분기 보고서가 없어 2016년 1~3분기 TTM 이 비고,
# 12개월 전 값과 비교하는 전략은 그 공백이 1년 뒤까지 번진다 → YoY 전략은 공백이 끝난 뒤 시작
SINCE_YOY = "2018-04-01"


def fin(name):
    return K.load(f"fin_{name}").reindex(columns=d.C.columns)


def _cap(dt):
    return d.CAP.reindex(dt)


def _rk(x, asc=True, worst=False):
    r = x.rank(axis=1, ascending=asc, pct=True)
    return r.fillna(1.0) if worst else r    # worst=True: 값 없음(적자 PER 등)은 꼴찌 순위로(종목 통째 배제 방지)


def _gpa(dt): return (fin("gp_ttm") / fin("assets")).reindex(dt)


def _pbr(dt):
    p = fund("PBR").reindex(dt)
    return p.where(p > 0)


def _shares_ok(dt):
    shares = d.CAP / d.C          # 수정주가 기준 주식수: 액면분할·무상증자는 증자로 보지 않음
    return (shares / shares.shift(250) <= 1.01).reindex(dt)


def _new_f(dt):
    """신F-스코어 3항목: 순이익>0, 영업현금흐름>0, 무증자."""
    return (fin("ni_ttm").reindex(dt) > 0) & (fin("ocf_ttm").reindex(dt) > 0) & _shares_ok(dt)


def _fscore(dt):
    """피오트로스키 F-score 9항목(연간 변화는 12개월 전 스냅샷 대비)."""
    ni, ocf, a = fin("ni_ttm"), fin("ocf_ttm"), fin("assets")
    liab, ca, cl = fin("liab"), fin("cur_assets"), fin("cur_liab")
    gp, rev = fin("gp_ttm"), fin("rev_ttm")
    roa = ni / a
    lev, cr = liab / a, ca / cl
    gm, at = gp / rev, rev / a
    lag = lambda x: x.shift(12)
    parts = [roa > 0, ocf > 0, roa > lag(roa), ocf > ni, lev < lag(lev), cr > lag(cr),
             gm > lag(gm), at > lag(at)]
    f = sum(p.astype(float) for p in parts).reindex(dt)
    return f + _shares_ok(dt).astype(float)


def _yearly(dt):
    return dt[dt >= pd.Timestamp(SINCE)]


@strat("LT39", "재무", "신마법공식 1.0(저PBR+고GP/A 순위합) 30종목, 연1회", "강환국 『할 수 있다! 퀀트 투자』, 인텔리퀀트 #573",
       since=SINCE)
def lt39():
    from strategies import _july
    dt = _yearly(_july())
    return RB(pick(_rk(_pbr(dt)) + _rk(_gpa(dt), False), d.U, 30, dt, ascending=True))


@strat("LT40", "재무", "신마법공식 분기판(저PBR+고GP/A) 30종목, 분기", "강환국 신마법공식 + 분기 리밸런싱", since=SINCE)
def lt40():
    dt = _yearly(_qe())
    return RB(pick(_rk(_pbr(dt)) + _rk(_gpa(dt), False), d.U, 30, dt, ascending=True))


@strat("LT41", "재무", "F-score(9항목)≥7 중 저PBR 30종목, 연1회", "Piotroski 2000, henryquant KOSPI F-score", since=SINCE)
def lt41():
    from strategies import _july
    dt = _yearly(_july())
    return RB(pick(_pbr(dt), (_fscore(dt) >= 7) & d.U.reindex(dt), 30, dt, ascending=True))


@strat("LT42", "재무", "신F-스코어(흑자·영업현금흐름+·무증자)+저PBR 25종목, 연1회", "강환국/인텔리퀀트 #583", since=SINCE)
def lt42():
    from strategies import _july
    dt = _yearly(_july())
    return RB(pick(_pbr(dt), _new_f(dt) & d.U.reindex(dt), 25, dt, ascending=True))


@strat("LT43", "재무", "영업현금흐름/시총(OCF/P) 상위 30, 분기", "JDQS 2020 한국 이상현상(가치 중 최강)", since=SINCE)
def lt43():
    dt = _yearly(_qe())
    return RB(pick(fin("ocf_ttm").reindex(dt) / _cap(dt), d.U, 30, dt))


@strat("LT44", "재무", "슈퍼퀄리티 업그레이드(신F 통과 → 고GP/A·저자산성장·영업이익성장·저변동성 순위합) 25종목, 연1회",
       "인텔리퀀트 #867(강환국 유튜브 503회)", since=SINCE, assume="영업이익/차입금 성장 대신 영업이익 성장률")
def lt44():
    from strategies import _july
    dt = _yearly(_july())
    a = fin("assets")
    ag = (a / a.shift(12) - 1).reindex(dt)
    op = fin("op_ttm")
    og = ((op - op.shift(12)) / op.shift(12).abs()).reindex(dt)
    vol = d.R.rolling(60, min_periods=40).std().reindex(dt)
    score = _rk(_gpa(dt), False) + _rk(ag) + _rk(og, False) + _rk(vol)
    return RB(pick(score, _new_f(dt) & d.U.reindex(dt), 25, dt, ascending=True))


@strat("LT45", "재무", "저자산성장 30종목(흑자 기업), 연1회", "회계정보연구 2016, 투자 팩터(한국)", since=SINCE)
def lt45():
    from strategies import _july
    dt = _yearly(_july())
    a = fin("assets")
    ag = (a / a.shift(12) - 1).reindex(dt)
    return RB(pick(ag, (fin("ni_ttm").reindex(dt) > 0) & d.U.reindex(dt), 30, dt, ascending=True))


@strat("LT46", "재무", "PIR(영업이익 TTM 증가분/시총) 상위 20, 분기", "인텔리퀀트 #981", since=SINCE_YOY)
def lt46():
    dt = _yearly(_qe())
    op = fin("op_ttm")
    return RB(pick((op - op.shift(12)).reindex(dt) / _cap(dt), d.U, 20, dt))


@strat("LT47", "재무", "GP/A 단독 상위 30, 분기", "Novy-Marx 2013, 김민기 외 2018 재무관리연구", since=SINCE)
def lt47():
    dt = _yearly(_qe())
    return RB(pick(_gpa(dt), d.U, 30, dt))


@strat("LT48", "재무", "밸류업 복합(저PER·저PBR·고ROE·영업현금흐름/시총·배당) 30종목, 분기",
       "하나증권 코리아밸류업150 시뮬레이션(현금흐름 포함판)", since=SINCE)
def lt48():
    dt = _yearly(_qe())
    per = fund("PER").reindex(dt)
    roe = (fin("ni_own_ttm").fillna(fin("ni_ttm")) / fin("equity_own").fillna(fin("equity"))).reindex(dt)  # PER·PBR 과 같은 지배주주 기준
    score = (_rk(per.where(per > 0)) + _rk(_pbr(dt)) + _rk(roe, False)
             + _rk(fin("ocf_ttm").reindex(dt) / _cap(dt), False) + _rk(fund("DIV").reindex(dt), False))
    return RB(pick(score, d.U, 30, dt, ascending=True))


def _multifactor(dt):
    """인텔리퀀트 #869: 가치(PER·PBR·PSR·PFCR) + 성장(영업이익·순이익 전년·전기 대비) + 소형 + 퀄리티(GP/A·저변동성) 1:1:1:1.
    전기 대비 = TTM 의 직전 분기 대비 변화(분기 단독 실적 대신)."""
    per = fund("PER").reindex(dt)
    cap = _cap(dt)
    psr = cap / fin("rev_ttm").reindex(dt)
    fcf = fin("ocf_ttm") - fin("capex_ttm").abs().fillna(0.0)       # 원문 PFCR: 잉여현금흐름
    pcr = cap / fcf.reindex(dt)
    W = dict(worst=True)   # 1:1:1:1 합산: 한 항목이 없다고 종목을 통째로 빼지 않고 그 항목만 꼴찌 처리
    value = (_rk(per.where(per > 0), **W) + _rk(_pbr(dt), **W) + _rk(psr.where(psr > 0), **W)
             + _rk(pcr.where(pcr > 0), **W)) / 4
    op, ni = fin("op_ttm"), fin("ni_ttm")
    gr = lambda x, k: ((x - x.shift(k)) / x.shift(k).abs()).reindex(dt)   # k=12 전년 대비, k=3 전기(분기) 대비
    growth = (_rk(gr(op, 12), False, **W) + _rk(gr(ni, 12), False, **W)
              + _rk(gr(op, 3), False, **W) + _rk(gr(ni, 3), False, **W)) / 4
    size = _rk(cap, **W)
    vol = d.R.rolling(63, min_periods=40).std().reindex(dt)
    quality = (_rk(_gpa(dt), False, **W) + _rk(vol, **W)) / 2
    return value + growth + size + quality


@strat("LT49", "재무", "멀티팩터(가치·성장·소형·퀄리티 1:1:1:1) 20종목, 월간", "인텔리퀀트 #869", since=SINCE_YOY,
       assume="원전의 거래대금 1억 필터 대신 유니버스(시총 1,000억 이상)")
def lt49():
    dt = _yearly(_me())
    return RB(pick(_multifactor(dt), d.U, 20, dt, ascending=True))


@strat("LT50", "재무", "멀티팩터 20 + 할로윈(5~10월 국고채10년)", "인텔리퀀트 #869 할로윈 결합", since=SINCE_YOY,
       assume="원전의 거래대금 1억 필터 대신 유니버스(시총 1,000억 이상). 결측 항목은 그 항목만 꼴찌 순위")
def lt50():
    dt = _yearly(_me())
    return _with_halloween(pick(_multifactor(dt), d.U, 20, dt, ascending=True), dt)
