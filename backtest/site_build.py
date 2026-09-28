"""전략 검증 결과 웹사이트 생성: results → JSON → site_template.html → site/index.html (+ robots.txt).

표시 목록·등급·배지는 results/site_strategies.csv(사용자 확정). 지표 정의는 engine.metrics 와 같고,
페이지의 selfcheck() 가 EXPECTED(파이썬 계산값)와 브라우저 계산을 대조한다.

월 갱신(수동, 추후 '월간 전략 사이트 갱신' 스킬):
    python collect.py && python kr_data.py        (시스템 python)
    .venv/Scripts/python run_all.py --jobs 2 && .venv/Scripts/python meta.py && .venv/Scripts/python report.py
    .venv/Scripts/python site_build.py            → site/index.html, site/robots.txt
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import engine as E
import kr_data as K
import report as R
import strategies as S
from site_desc import DESC, SHORT

ROOT = Path(__file__).parent
RES = ROOT / "results"
OUT = ROOT / "site"
TEMPLATE = ROOT / "site_template.html"
TAX = 0.154
BENCH = {"BM_KOSPI200": "코스피200 (KODEX200)", "BM_NASDAQ100": "나스닥100 (TIGER, 환노출)", "BM_SP500": "S&P500 (SPY, 원화 환산)"}
PERIODS = {"2011-2024": ("2011", "2024"), "2020-2026": ("2020", "2026")}

# 최종 독립 검증(scratchpad/verify_1~3.md) — (출처 등급, 퀀트 판정, 근거)
REVIEW = {
    "LT32": ("D", "주의", "원 규칙 특정 불가. 경제적 근거 있음, 6개 개별종목 중 가장 건전. 2023 배당절차 개선으로 효과 약화 가능"),
    "AA17": ("D", "주의", "출처가 블로그뿐. 나스닥100 환노출과 달러 노출이 겹침"),
    "AA09": ("C", "주의", "Keller HAA 로직 일치, 미국 신호를 총수익으로 수정함. 미국 채권 신호로 한국 채권 선택"),
    "AA10": ("B", "주의", "LAA 규칙 일치. 약세 신호가 16년 7회뿐, 타이밍 없는 정적 4자산도 비슷"),
    "AA02": ("B", "건전", "비중은 2차 자료로만 확인. 305080 상장 때문에 2018-10 시작"),
    "AA03": ("D", "주의", "20%/50% 비중이 출처에 없음. AA02 와 상관 0.94"),
    "AA06": ("C", "주의", "절대모멘텀 판정 방식이 원문과 다름(결과 동일). 2025 이후 의존 큼"),
    "AA01": ("A", "주의", "할로윈 규칙 일치. 캘린더 이상현상이라 경제적 근거 약함"),
    "LT01": ("C", "주의", "유니버스(시총 1,000억+) 안 하위 20%라 1,000~1,400억 구간만 남음. 신호 월 선택 운 큼"),
    "AA05": ("B", "건전", "주식·채권·금·달러 25% 일치. 분기 리밸런싱은 출처에 없음"),
    "AA16": ("-", "수정 후 재검증 전", "원문(12개월·필터 없음)으로 수정함"),
    "AA08": ("D", "주의", "3자산 변형 확인 불가. AA07 과 상관 0.97"),
    "AA15": ("C", "주의", "VAA 구조 일치, 신호를 총수익으로 수정함. 미국 채권 신호로 한국 채권 선택"),
    "LT21": ("-", "수정 후 재검증 전", "영업현금흐름 조건·수정주가 무증자 판정으로 수정함"),
    "AA12": ("D", "건전", "원 규칙 미확인, 파라미터 거의 없음"),
    "AA04": ("D", "주의", "사실상 채권 펀드(주식 6~32%)"),
    "AA20": ("B", "건전(약함)", "QuantConnect 규칙 일치, 성과 약함"),
    "AA19": ("B", "건전", "구성 원출처 확인, 연 1회 리밸런싱은 미확인"),
    "AA07": ("B", "주의", "Faber 10개월선 일치. 6~12개월 모두 2024까지 KODEX200 미만"),
    "AA13": ("A", "건전(약함)", "Faber 원문 PDF 일치, 초과 샤프 약함"),
    "DF03": ("B", "주의", "AllocateSmartly 규칙 일치. 달러·원유선물을 위험균등에 넣은 점 부적절"),
    "LT27": ("C", "문제", "할로윈 채권 구간이 12개 후보 중 1등(사후 선택), 여름 효과 t=-0.93"),
    "AA21": ("D", "문제", "국면 라벨 1개월 지연에 샤프 0.59→0.43, 무작위 라벨 범위 안"),
    "LT50": ("C", "문제", "기반 멀티팩터(LT49) 무우위, 할로윈 t=-0.90, 구간 12개 중 2등"),
    "AA18": ("B", "문제", "2024까지 KODEX200 수준, 조회기간에 민감"),
    "DF06": ("D", "문제", "2017 이전 KODEX200 과 동일, 방어 효과 없음"),
}
UNREVIEWED = ("-", "독립 검증 미실시", "최종 독립 검증 대상(당시 표시 후보) 밖")


CYCLE_OVERRIDE = {"LT32": "연 3회(9월 말·12월 말·1월 15일)"}
EXCL = ["ON05", "CD01", "ON07", "SW14", "SW15", "SW16", "LT14", "BM02"]   # 체결 불가·수급 결함·벤치마크
PROBLEM = ["AA18", "AA21", "DF01", "DF04", "DF06", "DF07", "LT27", "LT33", "LT50"]   # 퀀트 판정 '문제'


def cycle(name: str, sid: str = "") -> str:
    if sid in CYCLE_OVERRIDE:
        return CYCLE_OVERRIDE[sid]
    return "연 1회" if "연1회" in name.replace(" ", "") else "분기" if "분기" in name else "월간"


def build_tiers(write=True) -> pd.DataFrame:
    """사이트 등급(2026-09-25 사용자 확정 규칙). 판정 수치는 정확히 2011-01-01~2024-12-31.
    공통: 11~24 연수익 > 0, 앞 구간(IS) 연수익 > 0, 비용 2배 OOS 샤프 > 0. 단일 지수 보유(H*)·EXCL 제외.
    1 검증 통과: 11~24 초과 샤프 ≥ 0.5 & 알파 t > 2 / 2 경계: 샤프 ≥ 0.5 / 3: 수익률 또는 샤프가 코스피200 초과 /
    4: 코스피200 이하 & 샤프 > 0 / 주의(숨김): 코스피200 초과지만 퀀트 판정 '문제'."""
    m = pd.read_csv(RES / "metrics.csv").set_index("id")
    d = pd.read_parquet(RES / "daily.parquet")
    ab = R.alpha_beta(d)
    p24 = lambda s: E.metrics(d[s].dropna().loc["2011":"2024"])
    k = p24("H01")
    rows = {}
    for sid in m.index:
        if sid in EXCL or sid.startswith("H0") or sid not in d:
            continue
        x = p24(sid)
        c, s = x.get("CAGR", np.nan), x.get("Sharpe", np.nan)
        if not (c > 0 and m.at[sid, "IS_CAGR"] > 0 and m.at[sid, "cost2x_OOS_Sharpe"] > 0):
            continue
        t = ab.at[sid, "T24"] if sid in ab.index else np.nan
        beat = c > k["CAGR"] or s > k["Sharpe"]
        prob = sid in PROBLEM
        tier = ("주의(숨김)" if beat else None) if prob else (
            "1 검증 통과" if s >= 0.5 and t > 2 else "2 경계" if s >= 0.5 else
            "3 코스피200보다 나음" if beat else "4 수익 플러스·지수 이하" if s > 0 else None)
        if tier:
            rows[sid] = {"tier": tier, "cat": m.at[sid, "cat"], "name": m.at[sid, "name"], "C24": c, "S24": s, "T24": t,
                         "ALL_MDD": m.at[sid, "ALL_MDD"], "수익률>코스피200": c > k["CAGR"], "샤프>코스피200": s > k["Sharpe"],
                         "MDD 코스피200보다 깊음": m.at[sid, "ALL_MDD"] < m.at["H01", "ALL_MDD"]}
    out = pd.DataFrame(rows).T.sort_values(["tier", "S24"], ascending=[True, False])
    if write:
        out.to_csv(RES / "site_strategies.csv", encoding="utf-8-sig")
        print(out.tier.value_counts().sort_index().to_string())
    return out


def annual_tax(daily: pd.Series, rate=TAX) -> pd.Series:
    """과세 자산 100% 보유: 그해 누적 양(+) 수익 × 15.4% 를 그해 마지막 거래일에 차감(run_all.etf_tax 특수형)."""
    out = daily.copy()
    for _, g in daily.groupby(daily.index.year):
        gain = (1 + g).prod() - 1
        if gain > 0:
            out.loc[g.index[-1]] -= rate * gain
    return out


def load():
    site = pd.read_csv(RES / "site_strategies.csv", index_col=0)
    m = pd.read_csv(RES / "metrics.csv").set_index("id")
    daily = pd.read_parquet(RES / "daily.parquet")
    return {"site": site, "metrics": m, "daily": daily}


def benchmarks(daily: pd.DataFrame) -> pd.DataFrame:
    idx = daily.index
    us, lh = K.load("us"), K.load("longhist")
    spy = us["SPY"].reindex(idx.union(us.index)).ffill().reindex(idx)
    fx = lh["usdkrw"].reindex(idx.union(lh.index)).ffill().reindex(idx)
    sp = annual_tax((spy * fx).pct_change().dropna())
    return pd.DataFrame({"BM_KOSPI200": daily["H01"], "BM_NASDAQ100": daily["H05"], "BM_SP500": sp.reindex(idx)})


def _px_of(res):
    if res.get("px") is not None:
        return {k.lower(): v for k, v in res["px"].items()}
    return {k: K.load(f"adj_{k}") for k in ("open", "high", "low", "close")}


def holdings(sid):
    """체결까지 끝난 마지막 리밸런싱의 목표 비중(>0). 데이터 마지막 날에 찍힌 신호는 아직 체결 전이라 제외."""
    res = S.REG[sid]["fn"]()
    w = res["weights"]
    last_day = _px_of(res)["close"].index[-1]
    w = w[w.index < last_day].dropna(how="all")
    row = w.iloc[-1]
    row = row[row > 1e-9]
    meta = K.load("meta")
    name = lambda c: K.ETFS.get(c) or (meta.loc[c, "name"] if c in meta.index else c)
    h = pd.DataFrame({"code": row.index, "name": [name(c) for c in row.index], "weight": row.to_numpy()})
    return h.sort_values("weight", ascending=False).reset_index(drop=True), pd.Timestamp(w.index[-1])


def warn_reasons(codes, asof):
    """보유 개별종목의 부실 전조 사유(strategies.warn_mask 와 같은 조건)."""
    ev = pd.read_parquet(K.PANEL.parent / "dart" / "disclosures.parquet")
    ev = ev[ev.stock_code.isin(codes)].assign(dt=lambda x: pd.to_datetime(x.rcept_dt))
    ev = ev[(ev.dt <= asof) & (ev.dt > asof - pd.Timedelta(days=365))]
    ni = pd.read_parquet(K.PANEL / "fin_ni_ttm.parquet").loc[:asof]
    label = {"admin_issue": "관리종목", "unfaithful": "불성실공시", "audit_opinion": "감사의견 공시",
             "cb": "CB 발행", "bw": "BW 발행", "rights_offering": "유상증자"}
    out = {}
    for c in codes:
        e = ev[ev.stock_code == c]
        loss = c in ni and ni[c].dropna().size and ni[c].dropna().iloc[-1] < 0
        rs = [f"{label[k]}({e[e.event == k].dt.max():%Y-%m})" for k in S.WARN_A if (e.event == k).any()]
        if loss:
            rs += [f"적자+{label[k]}({e[e.event == k].dt.max():%Y-%m})" for k in S.WARN_B if (e.event == k).any()]
        if rs:
            out[c] = ", ".join(rs)
    return out


def _clean(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(x) if isinstance(x, (np.floating, float, int, np.integer)) else x


def _metrics(r):
    r = r.dropna()
    if len(r) < 60:
        return None
    m = E.metrics(r)
    return {k: _clean(m.get(k)) for k in ("CAGR", "Sharpe", "Sortino", "MDD", "Calmar")}


def payload():
    L = load()
    site, m, daily = L["site"], L["metrics"], L["daily"]
    ids = list(site.index)
    bm = benchmarks(daily)
    ser = pd.concat([daily[ids], bm], axis=1).loc["2010":].round(6)   # 페이지에 넣는 값 그대로(기대값도 같은 입력으로 계산)
    rf = E.rf_daily(ser.index)
    ab = R.alpha_beta(pd.concat([daily, bm], axis=1))
    dsr = R.deflated_sharpe(daily)
    meta_ = {}
    for sid in ids:
        info, s = S.REG[sid], site.loc[sid]
        h, last = holdings(sid)
        stocks = [c for c in h.code if c not in K.ETFS]
        warn = warn_reasons(stocks, ser.index[-1]) if stocks else {}
        g, v, why = REVIEW.get(sid, UNREVIEWED)
        mm = m.loc[sid]
        meta_[sid] = {
            "name": info["name"], "cat": info["cat"], "tier": s["tier"], "src": info["src"], "assume": info["assume"], "desc": DESC.get(sid), "short": SHORT.get(sid),
            "badges": {"cagr": bool(s["수익률>코스피200"]), "sharpe": bool(s["샤프>코스피200"]), "mdd": bool(s["MDD 코스피200보다 깊음"])},
            "review": {"grade": g, "verdict": v, "why": why}, "cycle": cycle(info["name"], sid), "last_rebal": f"{last:%Y-%m-%d}",
            "holdings": [{"code": r.code, "name": r.name, "w": round(float(r.weight), 4), "warn": warn.get(r.code)}
                         for r in h.itertuples()],
            "fixed": {k: _clean(v_) for k, v_ in {
                "win": mm.get("ALL_WinRate"), "payoff": mm.get("ALL_Payoff"), "pf": mm.get("ALL_PF"),
                "trades": mm.get("ALL_Trades"), "turnover": mm.get("Turnover_rb", mm.get("OOS_Turnover")),
                "cost2x": mm.get("cost2x_OOS_Sharpe"), "dsr": dsr.get(sid)}.items()},
        }
    for b, nm in BENCH.items():
        meta_[b] = {"name": nm, "cat": "벤치마크", "tier": "벤치마크", "fixed": {}, "holdings": []}
    for k in list(ids) + list(BENCH):
        a = ab.loc[k] if k in ab.index else {}
        meta_[k].setdefault("fixed", {}).update({"alpha": _clean(a.get("A24")), "alpha_t": _clean(a.get("T24")),
                                                  "beta": _clean(a.get("B24"))})
    expected = {k: {p: _metrics(ser[k].loc[a:b]) for p, (a, b) in PERIODS.items()} for k in ser}
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "asof": f"{ser.index[-1]:%Y-%m-%d}",
        "dates": [f"{d:%Y-%m-%d}" for d in ser.index],
        "rf": [round(float(x), 9) for x in rf],
        "series": {k: [None if not np.isfinite(x) else round(float(x), 6) for x in ser[k].to_numpy()] for k in ser},
        "meta": meta_, "expected": expected, "bench": list(BENCH),
    }


def render(p):
    OUT.mkdir(exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(p, ensure_ascii=False, separators=(",", ":")).replace("</", "<\/"))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / "robots.txt").write_text("User-agent: *\nDisallow: /\n", encoding="utf-8")
    print(f"저장: {OUT / 'index.html'} ({(OUT / 'index.html').stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    build_tiers()
    render(payload())
