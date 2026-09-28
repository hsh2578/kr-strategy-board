"""3차 조사 전략(2026-09-24): 하락·위기 방어 ETF, 단순 보유 자산, 한국 재현 이상현상, 감마 거시 배분.

- 방어 전략은 월말 신호 → 다음 거래일 시가 리밸런싱(엔진 rebalance). KRX 리스크컨트롤 지수는 매일 조정하지만
  비용을 줄이려 월 1회로 단순화했다(ponytail: 일간 조정은 필요할 때 추가).
- 단순 보유(H**)는 메타 시스템이 국면별로 자산 자체를 고를 수 있게 하는 후보다.
- 이상현상 정의는 Han·Lee·Kang(2020, JDQS) 한국 재현 목록을 우리 데이터 필드로 옮긴 것이다.
"""
import numpy as np
import pandas as pd

import kr_data as K
from strategies import RB, _me, _qe, d, etf_px, fund, ma, month_ends, pick, strat
from strategies2 import _etf_alloc, _kr_me, _at
from strategies3 import fin

CASH = "153130"


def _vol(codes, n):
    c = etf_px(*codes)["close"]
    return c.pct_change().rolling(n, min_periods=int(n * 0.8)).std() * np.sqrt(252)


def _asof(frame, t):
    x = frame.loc[:t]
    return x.iloc[-1] if len(x) else None


# ═══════════════ 단순 보유 (메타 시스템 후보 자산) ═══════════════
def _hold(code):
    return _etf_alloc([code], lambda t: {code: 1.0})   # 월말마다 100% 목표(비중 불변 → 추가 매매 없음), 상장 직후부터 보유


for _sid, _code, _nm in [("H01", "069500", "KODEX200"), ("H02", "148070", "국고채10년"), ("H03", "132030", "금선물"),
                         ("H04", "138230", "미국달러선물"), ("H05", "133690", "미국나스닥100"),
                         ("H06", "114800", "KOSPI200 인버스"), ("H07", "289480", "KOSPI200 커버드콜"),
                         ("H08", "161510", "고배당주"), ("H09", "114260", "국고채3년")]:
    strat(_sid, "자산보유", f"{_nm} 보유", "메타 시스템 후보 자산(단순 보유)", etf=True)(
        (lambda c: (lambda: _hold(c)))(_code))


@strat("BM02", "벤치마크", "KODEX200 45% + 단기채 55% 고정, 월간", "비교 기준(검증에서 방어 전략들이 못 이긴 정적 조합)", etf=True)
def bm02(): return _etf_alloc(["069500", CASH], lambda t: {"069500": 0.45, CASH: 0.55})


# ═══════════════ 하락·위기 방어 ═══════════════
@strat("DF01", "방어", "실무형 변동성 타기팅(KODEX200, 목표 10%, max(20·60일 변동성), 레버리지 없음, 월간)",
       "변동성 타기팅(Moreira·Muir 2017 은 레버리지 허용 원형 — 롱온리·무레버리지 조건에 맞춘 실무형)",
       etf=True, assume="원 논문과 달리 비중 상한 100%(레버리지 없음), 20·60일 중 큰 변동성(보수적), 일간 대신 월말 조정")
def df01():
    v = np.maximum(_vol(["069500"], 20)["069500"], _vol(["069500"], 60)["069500"])
    def fn(t):
        x = _asof(v, t)
        if x is None or np.isnan(x):
            return {}
        w = min(0.10 / x, 1.0)
        return {"069500": w, CASH: 1 - w}
    return _etf_alloc(["069500", CASH], fn)


TS_ASSETS = ["069500", "148070", "132030", "138230", "130680"]


@strat("DF02", "방어", "멀티에셋 시계열 모멘텀(12개월 부호·변동성 스케일, 5자산)", "Moskowitz·Ooi·Pedersen 2012 TSMOM", etf=True,
       assume="숏 없음(음수면 현금), 자산당 비중 = 1/5 × min(1, 10%/60일 변동성)")
def df02():
    m = _kr_me(*TS_ASSETS)
    r12 = m / m.shift(12) - 1
    vol = _vol(TS_ASSETS, 60)
    def fn(t):
        r, v = _at(r12, t), _asof(vol, t)
        if r is None or v is None:
            return {}
        w = {c: 0.2 * min(1.0, 0.10 / v[c]) for c in TS_ASSETS if pd.notna(r[c]) and r[c] > 0 and v[c] > 0}
        w[CASH] = 1 - sum(w.values())
        return w
    return _etf_alloc(TS_ASSETS + [CASH], fn)


@strat("DF03", "방어", "위험균등 + 10개월선 필터(주식·채권·금·달러·원유)", "AllocateSmartly 'The Trend is Our Friend'", etf=True)
def df03():
    m = _kr_me(*TS_ASSETS)
    ok = m > m.rolling(10).mean()
    vol = _vol(TS_ASSETS, 252)
    def fn(t):
        o, v = _at(ok, t), _asof(vol, t)
        if o is None or v is None or v.isna().all():
            return {}
        iv = (1 / v).where(v > 0)
        base = iv / iv.sum()
        w = {c: float(base[c]) for c in TS_ASSETS if pd.notna(base[c]) and bool(o[c])}
        w[CASH] = 1 - sum(w.values())
        return w
    return _etf_alloc(TS_ASSETS + [CASH], fn)


@strat("DF04", "방어", "KODEX200 70% + 달러선물 30%(원달러 10개월선 위일 때만, 아니면 단기채)", "토스증권 리서치(주식-달러 상관 -0.07)", etf=True)
def df04():
    m = _kr_me("138230")["138230"]
    up = m > m.rolling(10).mean()
    return _etf_alloc(["069500", "138230", CASH],
                      lambda t: {"069500": 0.7, ("138230" if bool(_at(up, t)) else CASH): 0.3})


@strat("DF05", "방어", "경기방어 섹터 로테이션(KOSPI 10개월선 아래면 필수소비재·헬스케어, 위면 KODEX200)",
       "2026.7 급락기 섹터 성과(FnGuide), 경기방어 로테이션", etf=True, assume="방어 ETF는 3개월 모멘텀 양수일 때만, 아니면 단기채")
def df05():
    codes = ["069500", "266410", "227540"]
    m = _kr_me(*codes)
    bull = m["069500"] > m["069500"].rolling(10).mean()
    r3 = m / m.shift(3) - 1
    def fn(t):
        if bool(_at(bull, t)):
            return {"069500": 1.0}
        r = _at(r3, t)
        w = {c: 0.5 for c in ("266410", "227540") if r is not None and pd.notna(r[c]) and r[c] > 0}
        w[CASH] = 1 - sum(w.values())
        return w
    return _etf_alloc(codes + [CASH], fn)


@strat("DF06", "방어", "최소변동성 스위칭(KOSPI 10개월선 아래면 최소변동성 ETF)", "저변동성 이상현상(고봉찬·김진우)", etf=True)
def df06():
    m = _kr_me("069500")["069500"]
    bull = m > m.rolling(10).mean()
    listed = etf_px("279540")["close"]["279540"].first_valid_index()   # 최소변동성 ETF 상장 전엔 KODEX200
    return _etf_alloc(["069500", "279540"],
                      lambda t: {"069500": 1.0} if bool(_at(bull, t)) or pd.Timestamp(t) < listed
                      else {"279540": 1.0})


@strat("DF07", "방어", "TIPP 낙폭 통제(하한 = 최고가치×85%, 승수 4, KODEX200/단기채)", "이준행 외 2005 KCI TIPP", etf=True,
       assume="월말 조정, 비중 계산용 가치는 비용 전 수익률로 추적")
def df07():
    p = etf_px("069500", CASH)["close"].dropna()
    me = month_ends(p.index)
    rets = p.pct_change().fillna(0)
    val, peak, w_eq, rows = 1.0, 1.0, 0.0, {}
    prev = None
    for t in me:
        if prev is not None:
            seg = rets.loc[prev:t].iloc[1:]
            val *= float(np.prod(1 + w_eq * seg["069500"] + (1 - w_eq) * seg[CASH]))
        peak = max(peak, val)
        w_eq = float(np.clip(4 * (val - 0.85 * peak) / val, 0, 1))
        rows[t] = {"069500": w_eq, CASH: 1 - w_eq}
        prev = t
    return RB(pd.DataFrame(rows).T, etf=True, px=etf_px("069500", CASH))


@strat("DF08", "방어", "금 추세(10개월선 위면 금선물, 아니면 단기채)", "Baur·Lucey 2010 안전자산", etf=True)
def df08():
    m = _kr_me("132030")["132030"]
    up = m > m.rolling(10).mean()
    return _etf_alloc(["132030", CASH], lambda t: {("132030" if bool(_at(up, t)) else CASH): 1.0})


@strat("AA21", "자산배분", "감마팀 거시 국면 배분(골디락스 KODEX200 / 둔화 60:40 / 과열 원유 / 스태그 금)",
       "감마팀 발표(국면 기반 동적 자산배분), 국내 ETF판", etf=True, assume="원자재 = TIGER 원유선물, 채권 = 국고채10년")
def aa21():
    import regimes as RG
    lab = RG.load()["d2"].dropna()
    table = {"골디락스": {"069500": 1.0}, "둔화": {"069500": 0.6, "148070": 0.4},
             "과열": {"130680": 1.0}, "스태그플레이션": {"132030": 1.0}}
    def fn(t):
        p = pd.Timestamp(t).to_period("M")
        return table.get(lab.get(p)) if p in lab.index else {}
    return _etf_alloc(["069500", "148070", "130680", "132030"], fn)


# ═══════════════ 한국 재현 이상현상 (Han·Lee·Kang 2020 JDQS 등) ═══════════════
SINCE_FIN = "2016-05-01"


def _yr(dt):
    return dt[dt >= pd.Timestamp(SINCE_FIN)]


@strat("AN01", "이상현상", "비율발생액(Pta) 낮은 30종목, 연1회", "JDQS 2020 Pta t=4.13, 한국 percent accruals 연구", since=SINCE_FIN,
       assume="Pta ≈ (순이익-영업현금흐름)/|순이익| (TTM)")
def an01():
    from strategies import _july
    dt = _yr(_july())
    ni, ocf = fin("ni_ttm"), fin("ocf_ttm")
    pta = ((ni - ocf) / ni.abs().replace(0, np.nan)).reindex(dt)
    return RB(pick(pta, d.U, 30, dt, ascending=True))


@strat("AN02", "이상현상", "복합주식발행(Cei) 낮은 30종목(주식수 증가 기업 회피), 연1회", "JDQS 2020 Cei t=3.36, Daniel·Titman")
def an02():
    from strategies import _july
    dt = _july()
    cei = (np.log(d.CAP / d.CAP.shift(1260)) - np.log(d.C / d.C.shift(1260))).reindex(dt)
    return RB(pick(cei, d.U, 30, dt, ascending=True))


def _mkt_ret():
    return etf_px("069500")["close"]["069500"].pct_change()


@strat("AN03", "이상현상", "잔차모멘텀 6개월(36개월 CAPM 잔차) 상위 30, 월간", "JDQS 2020 t=3.29, Blitz·Huij·Martens")
def an03():
    dt = _me()
    mp = d.C.reindex(dt)
    r = mp.pct_change()
    km = etf_px("069500")["close"]["069500"].reindex(dt).pct_change()
    rows = {}
    for i in range(37, len(dt)):
        R = r.iloc[i - 36:i]            # t 이전 36개월(최근 1개월 제외)
        M = km.iloc[i - 36:i]
        ok = R.notna().sum() >= 30
        Mc = M - M.mean()
        beta = (R.sub(R.mean())).mul(Mc, axis=0).sum() / (Mc ** 2).sum()
        alpha = R.mean() - beta * M.mean()
        res = R - alpha - np.outer(M, beta)
        last6 = res.iloc[-6:].sum()
        rows[dt[i]] = (last6 / res.std()).where(ok)
    score = pd.DataFrame(rows).T.reindex(dt)
    return RB(pick(score, d.U, 30, dt))


@strat("AN04", "이상현상", "고유변동성(1개월 CAPM 잔차) 낮은 30종목, 월간", "JDQS 2020 Ivc_1 t=4.27, Kang·Lee·Sim 2014")
def an04():
    dt = _me()
    m = _mkt_ret().reindex(d.R.index)
    var_m = m.rolling(21, min_periods=15).var()
    roll = lambda x: x.rolling(21, min_periods=15).mean()
    cov = roll(d.R.mul(m, axis=0)) - roll(d.R).mul(roll(m), axis=0)
    beta = cov.div(var_m, axis=0)
    ivol = (d.R.rolling(21, min_periods=15).var() - beta.pow(2).mul(var_m, axis=0)).clip(lower=0) ** 0.5
    return RB(pick(ivol.reindex(dt), d.U, 30, dt, ascending=True))


@strat("AN05", "이상현상", "매출/시총(Sp) 상위 30, 분기", "JDQS 2020 Sp t=3.55", since=SINCE_FIN)
def an05():
    dt = _yr(_qe())
    return RB(pick(fin("rev_ttm").reindex(dt) / d.CAP.reindex(dt), d.U, 30, dt))


@strat("AN06", "이상현상", "자산/시총(Am) 상위 30, 분기", "JDQS 2020 Am t=4.05", since=SINCE_FIN)
def an06():
    dt = _yr(_qe())
    return RB(pick(fin("assets").reindex(dt) / d.CAP.reindex(dt), d.U, 30, dt))


@strat("AN07", "이상현상", "좌측꼬리위험(12개월 일간수익률 2.5% 분위) 작은 30종목, 월간", "Eom·Eom·Park 2022, JDQS 개인투자자 좌측꼬리")
def an07():
    dt = _me()
    rows = {}
    for t in dt:
        i = d.R.index.get_loc(t)
        if i < 250:
            continue
        w = d.R.iloc[i - 249:i + 1]
        rows[t] = w.quantile(0.025).where(w.notna().sum() >= 200)
    var = pd.DataFrame(rows).T.reindex(dt)
    return RB(pick(var, d.U, 30, dt))         # 분위수가 클수록(덜 음수) 꼬리위험 작음


@strat("AN08", "이상현상", "영업이익/전기총자산(Ola) 상위 30, 분기", "JDQS 2020 Ola t=2.19(한국 수익성 중 유일 유의)", since=SINCE_FIN)
def an08():
    dt = _yr(_qe())
    ola = (fin("op_ttm") / fin("assets").shift(12)).reindex(dt)
    return RB(pick(ola, d.U, 30, dt))


@strat("AN09", "이상현상", "자본증가율(dBe)·영업발생액(Oa) 낮은 30종목, 연1회", "JDQS 2020 dBe t=2.53, Oa t=2.83", since=SINCE_FIN)
def an09():
    from strategies import _july
    dt = _yr(_july())
    eq, a = fin("equity"), fin("assets")
    dbe = (eq / eq.shift(12) - 1).reindex(dt)
    oa = ((fin("ni_ttm") - fin("ocf_ttm")) / ((a + a.shift(12)) / 2)).reindex(dt)
    score = dbe.rank(axis=1, pct=True) + oa.rank(axis=1, pct=True)
    return RB(pick(score, d.U, 30, dt, ascending=True))


@strat("AN10", "이상현상", "거래대금 변동계수(12개월) 낮은 30종목, 월간", "JDQS 2020 Cvd_12 t=2.63")
def an10():
    dt = _me()
    mtv = d.TV.groupby(d.TV.index.to_period("M")).sum()
    cv = (mtv.rolling(12, min_periods=10).std() / mtv.rolling(12, min_periods=10).mean())
    cv.index = month_ends(d.TV.index)
    return RB(pick(cv.reindex(dt), d.U, 30, dt, ascending=True))
