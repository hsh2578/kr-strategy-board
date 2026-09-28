"""국면 정의 4개(사전 등록). 모든 라벨은 월말 t 에 t 까지의 데이터만으로 계산한다(재라벨링 없음).

d1 가격: KODEX200 추세(월말 > 10개월 평균) × 변동성(60일 실현변동성 > 그때까지 월말값의 확장 중앙값) → 4국면
d2 거시(감마팀): 성장·물가 z-score 합성 + 발표 시차 + 히스테리시스 ±0.2 → 4국면
d3 점프모형(jumpmodels): KOSPI 일간수익률 EWM 특징 9개, λ=50 고정, 매월 expanding 재적합, predict_online 마지막 값 → 2국면
d4 난기류: KOSPI·원달러·금 일간수익률의 마할라노비스 거리(과거 5년 μ·Σ), 월평균 > 확장 75% 분위 → 2국면

데이터 이력 때문에 계획 대비 조정(RESUME.md·보고서에 기록):
 - d3 특징은 KODEX200 대신 KOSPI 지수(1997~)로 계산해 최소 8년 학습 후 2005년부터 라벨을 낸다.
 - d4 는 국고채 ETF 이력이 짧아 KOSPI·원달러·금 3자산으로 계산한다.
 - d2 의 GDP 는 분기말 + 2개월부터 사용한다.

    python regimes.py     # data/panel/regimes.parquet 생성
"""
import numpy as np
import pandas as pd

import kr_data as K

OUT = K.PANEL / "regimes.parquet"


def _me(idx):
    s = idx.to_series()
    return pd.DatetimeIndex(s.groupby(idx.to_period("M")).last())


# ─── d1 가격 ───
def d1_price():
    c = K.load("etf")["069500"]["Close"].dropna()
    me = _me(c.index)
    mc = c.loc[me]
    up = mc > mc.rolling(10).mean()
    vol = c.pct_change().rolling(60).std().loc[me]
    hi = vol > vol.expanding().median()
    lab = np.select([up & ~hi, up & hi, ~up & ~hi, ~up & hi], ["안정상승", "과열급등", "횡보둔화", "급락위기"], None)
    s = pd.Series(lab, index=me)
    return s[mc.rolling(10).count().eq(10) & vol.notna()]


# ─── d2 거시 ───
def _z(x, n):
    return (x - x.rolling(n, min_periods=n // 2).mean()) / x.rolling(n, min_periods=n // 2).std()


def d2_macro(th=0.2):
    m = K.load("macro_raw")
    me = pd.date_range("2005-01-31", m.index.max() + pd.offsets.MonthEnd(2), freq="ME")
    yoy = lambda s, k: s / s.shift(k) - 1
    monthly = lambda s: s.reindex(me.union(s.index)).ffill().reindex(me)
    lag = lambda s, k: monthly(s).shift(k)          # k개월 전 값이 월말 t 에 알려짐
    cpi, ppi = lag(yoy(m["cpi"].dropna(), 12), 2), lag(yoy(m["ppi"].dropna(), 12), 2)
    imp, exp_ = lag(yoy(m["import_px"].dropna(), 12), 1), lag(yoy(m["export"].dropna(), 12), 1)
    gdp = yoy(m["gdp"].dropna(), 4)
    gdp = monthly(gdp.set_axis(gdp.index + pd.offsets.MonthEnd(2)))   # 분기말 + 2개월 공표
    bsi = monthly(m["bsi"].dropna())
    g = (_z(gdp, 72) + _z(exp_, 36) + _z(bsi, 24)) / 3
    i = (_z(cpi, 36) + _z(ppi, 36) + _z(imp, 36)) / 3

    def hyst(x):
        st, out = np.nan, []
        for v in x:
            if np.isnan(v):
                out.append(np.nan); continue
            if np.isnan(st):
                st = 1.0 if v >= 0 else -1.0
            elif v > th:
                st = 1.0
            elif v < -th:
                st = -1.0
            out.append(st)
        return pd.Series(out, index=x.index)

    gs, is_ = hyst(g), hyst(i)
    lab = np.select([(gs > 0) & (is_ < 0), (gs > 0) & (is_ > 0), (gs < 0) & (is_ > 0), (gs < 0) & (is_ < 0)],
                    ["골디락스", "과열", "스태그플레이션", "둔화"], None)
    s = pd.Series(lab, index=me)
    return s.dropna()


# ─── d3 점프모형 ───
def _jm_features(ret):
    f = {}
    for hl in (5, 20, 60):
        m = ret.ewm(halflife=hl).mean()
        dd = np.sqrt((ret.clip(upper=0) ** 2).ewm(halflife=hl).mean())
        f[f"ret_{hl}"] = m
        f[f"dd_log_{hl}"] = np.log(dd)
        f[f"sortino_{hl}"] = m / dd
    return pd.DataFrame(f).replace([np.inf, -np.inf], np.nan).dropna()


def d3_jump(min_years=8, lam=50.0):
    from jumpmodels.jump import JumpModel
    from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
    px = K.load("longhist")["kospi"].dropna()
    ret = np.log(px).diff().dropna()
    X_all = _jm_features(ret)
    me = _me(X_all.index)
    me = me[me >= X_all.index[0] + pd.DateOffset(years=min_years)]
    out = {}
    for t in me:
        X = X_all.loc[:t]
        clip, scale = DataClipperStd(mul=3.0), StandardScalerPD()
        Xp = scale.fit_transform(clip.fit_transform(X))
        jm = JumpModel(n_components=2, jump_penalty=lam, cont=False, n_init=10, random_state=0)
        jm.fit(Xp, ret.loc[Xp.index], sort_by="cumret")
        out[t] = int(jm.predict_online(Xp).iloc[-1])
    s = pd.Series(out)
    return s.map({0: "강세", 1: "약세"})          # sort_by=cumret: 0 = 누적수익 높은 상태


# ─── d4 난기류 ───
def d4_turbulence(window=1260, q=0.75):
    lh = K.load("longhist")[["kospi", "usdkrw", "gold"]].dropna()
    r = np.log(lh).diff().dropna()
    vals = r.to_numpy()
    turb = np.full(len(r), np.nan)
    for i in range(window, len(r)):
        hist = vals[i - window:i]                     # t 이전 5년
        mu, cov = hist.mean(0), np.cov(hist, rowvar=False)
        d = vals[i] - mu
        turb[i] = d @ np.linalg.solve(cov, d)
    ts = pd.Series(turb, index=r.index).dropna()
    mm = ts.groupby(ts.index.to_period("M")).mean()
    mm.index = _me(ts.index)
    hi = mm > mm.expanding(min_periods=24).quantile(q)
    return hi[mm.expanding(min_periods=24).count() >= 24].map({True: "난기류", False: "평온"})


def build():
    regs = {"d1": d1_price(), "d2": d2_macro(), "d3": d3_jump(), "d4": d4_turbulence()}
    # 월 단위로 통일(거래일 월말 vs 달력 월말 차이 제거), 미래 달 제거
    now = pd.Timestamp.today().to_period("M")
    regs = {k: v.set_axis(v.index.to_period("M")).loc[:now] for k, v in regs.items()}
    df = pd.DataFrame(regs)
    df.index.name = "month"
    out = df.copy()
    out.index = out.index.to_timestamp(how="end").normalize()
    out.to_parquet(OUT)
    for k in ("d1", "d2", "d3", "d4"):
        s = df[k].dropna()
        runs = (s != s.shift()).cumsum().value_counts()
        print(f"{k}: {s.index[0]}~{s.index[-1]} · 국면별 개월수 {s.value_counts().to_dict()} · 평균 지속 {runs.mean():.1f}개월")
    return df


def load():
    """월(Period) 인덱스의 국면 라벨 표."""
    df = pd.read_parquet(OUT)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


if __name__ == "__main__":
    build()
