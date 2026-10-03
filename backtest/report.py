"""results/metrics.csv → 순위 산출 + Word 보고서(전략_백테스트_순위.docx)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_ORIENT  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

import kr_data as K  # noqa: E402

ROOT = Path(__file__).parent
RES = ROOT / "results"
OUT = ROOT.parent / "전략_백테스트_순위_v11.docx"
FONT = "Malgun Gothic"
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False
# dataviz 참조 팔레트(라이트): 범주 1~5, 텍스트·축 잉크
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MIN_OOS_TRADES = 30
# 백테스트상 체결을 가정했지만 실제로는 체결이 거의 불가능한 전략(순위에서 분리)
UNREALISTIC = {"ON05": "상한가 마감가에 매수 잔량이 쌓여 있어 실제 체결이 거의 안 됨"}

EXCLUDED = [
    ("30분봉 장중 모멘텀 (KOSPI 논문, systrader79)", "과거 분봉 없음 — KIS 최근 분봉으로 보조 검증 예정"),
    ("전일 동시간대 주기성 롱숏", "10분봉 필요 + 논문 저자가 거래세 넘지 못함을 인정"),
    ("시가범위돌파(ORB) / VWAP 눌림", "분봉 필요 · 하루 다회 매매(사용자 제외 기준)"),
    ("5분봉 변동성돌파 조건식", "분봉 필요 · 사용자 제외"),
    ("동시호가 예상체결 조건식(원형)", "예상체결량 데이터 없음 → CD04 시가갭으로 근사"),
    ("야간선물 신호 오버나잇", "야간선물 이력 없음"),
    ("이익 모멘텀(컨센서스)", "컨센서스 이력 없음"),
]


def score(df):
    """OOS 기준 종합점수 = OOS 샤프·칼마·PF·거래당 기대수익 백분위 평균."""
    s = pd.concat([df["OOS_Sharpe"].rank(pct=True), df["OOS_Calmar"].rank(pct=True),
                   df["OOS_PF"].rank(pct=True), df["OOS_AvgTrade"].rank(pct=True)], axis=1)
    return s.mean(axis=1)


def deflated_sharpe(daily: pd.DataFrame) -> pd.Series:
    """Deflated Sharpe Ratio(Bailey·López de Prado 2014): 시험한 전략 수(N)만큼 우연히 나올 최대 샤프를 뺀 뒤
    '진짜 샤프 > 0' 일 확률. 수익률은 단기채 대비 초과, N = daily 의 전략 수."""
    from scipy.stats import kurtosis, norm, skew
    import engine as E
    ex = daily.sub(E.rf_daily(daily.index), axis=0)
    sr = ex.mean() / ex.std()
    n, g = daily.shape[1], 0.5772156649
    emax = (1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e))   # N개 무실력 전략의 최대값 배수
    out = {}
    for c in ex:
        x = ex[c].dropna()
        if len(x) < 60 or x.std() == 0:
            continue
        sr0 = emax / np.sqrt(len(x) - 1)     # 귀무(샤프 0) 추정 분산 1/(T-1) — 전략 간 분산은 극단 손실 전략에 휘둘려 쓰지 않음
        z = (sr[c] - sr0) * np.sqrt(len(x) - 1) / np.sqrt(1 - skew(x) * sr[c] + (kurtosis(x, fisher=False) - 1) / 4 * sr[c] ** 2)
        out[c] = norm.cdf(z)
    return pd.Series(out)


def alpha_beta(daily: pd.DataFrame, end="2024-12-31", start="2011-01-01") -> pd.DataFrame:
    """KODEX200(H01) 초과수익에 대한 회귀: 연율 알파·베타·알파 t값(start~end). 시장이 오른 덕(베타)을 뺀 나머지."""
    import engine as E
    rf = E.rf_daily(daily.index)
    mk = daily["H01"] - rf
    out = {}
    for c in daily:
        z = pd.concat([daily[c] - rf, mk], axis=1).loc[start:end].dropna()
        if (daily[c].loc[z.index] != 0).sum() < 250:
            continue
        z = z.loc[daily[c].loc[z.index].ne(0).idxmax():]
        X = np.c_[np.ones(len(z)), z.iloc[:, 1].to_numpy()]
        y = z.iloc[:, 0].to_numpy()
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        se = np.sqrt(e.var(ddof=2) * np.linalg.inv(X.T @ X)[0, 0])
        out[c] = {"A24": b[0] * 252, "B24": b[1], "T24": b[0] / se}
    return pd.DataFrame(out).T


def load():
    m = pd.read_csv(RES / "metrics.csv")
    dly = pd.read_parquet(RES / "daily.parquet")
    import engine as E   # 11~24 수치는 정확히 2011-01-01~2024-12-31 (run_all 의 TO24 는 전략 시작일부터라 2010 이 섞임)
    for i, sid in enumerate(m["id"]):
        x = dly[sid].dropna().loc["2011":"2024"] if sid in dly else pd.Series(dtype=float)
        mm = E.metrics(x) if len(x) >= 60 else {}
        for k in ("CAGR", "Sharpe", "MDD"):
            m.loc[i, f"TO24_{k}"] = mm.get(k, np.nan)
    m["DSR"] = m["id"].map(deflated_sharpe(dly))
    ab = alpha_beta(dly)
    for k in ab:
        m[k] = m["id"].map(ab[k])
    r2 = dly.loc["2025":]
    m["R2526"] = m["id"].map((1 + r2).prod() ** (252 / max(len(r2), 1)) - 1)   # 2025~26 연율
    m["Turnover"] = m["OOS_Turnover"].fillna(m.get("Turnover_rb"))
    m["표본부족"] = (m["OOS_Trades"].fillna(0) < MIN_OOS_TRADES) | m["id"].isin(UNREALISTIC)
    m["IS·OOS 둘다+"] = (m["IS_CAGR"].fillna(m["OOS_CAGR"]) > 0) & (m["OOS_CAGR"] > 0)
    m["비용2배 생존"] = m["cost2x_OOS_Sharpe"] > 0
    intraday = ["ON08", "ON09", "DT02", "CD04", "SW10"]   # 랜덤 대조군 계산 불가(지정가 체결)
    m.loc[m["id"].isin(intraday), "random_OOS_Sharpe"] = np.nan
    if "random_OOS_Sharpe" in m:
        m["랜덤 대비 우위"] = np.where(m["random_OOS_Sharpe"].isna(), np.nan,
                                  m["OOS_Sharpe"] > m["random_OOS_Sharpe"])
    ok = ~m["표본부족"]
    m.loc[ok, "점수"] = m.loc[ok, "TO24_Sharpe"]            # 순위 = 2011~2024 초과 샤프(2025~26 급등장 제외)
    m = m.sort_values(["표본부족", "점수"], ascending=[True, False]).reset_index(drop=True)
    m["순위"] = np.nan
    m.loc[~m["표본부족"], "순위"] = np.arange(1, (~m["표본부족"]).sum() + 1)
    return m


def sensitivity(m):
    """results_slip_low / results_slip_zero 가 있으면 기존 결과와 합쳐 '낮춤' 샤프 순으로 정렬."""
    out = m[["id", "name", "OOS_CAGR", "OOS_Sharpe", "Turnover", "etf"]].copy()
    for tag, pre in (("slip_low", "low"), ("slip_zero", "zero")):
        f = ROOT / f"results_{tag}" / "metrics.csv"
        if not f.exists():
            return None
        s = pd.read_csv(f).set_index("id")
        for col in ("OOS_CAGR", "OOS_Sharpe", "IS_CAGR"):
            out[f"{pre}_{col}"] = out["id"].map(s[col]) if col in s else np.nan
    return out[~out["etf"]].sort_values("low_OOS_Sharpe", ascending=False)


def pct(x, d=1):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "" if pd.isna(x) or np.isinf(x) else f"{x:.{d}f}"


def flag(r):
    f = []
    f.append("견고" if r["IS·OOS 둘다+"] else "")
    f.append("비용OK" if r["비용2배 생존"] else "비용X")
    if r.get("랜덤 대비 우위") == 1.0:
        f.append("랜덤↑")
    elif r.get("랜덤 대비 우위") == 0.0:
        f.append("랜덤↓")
    return " ".join(x for x in f if x)


# ─── 차트 ───
def chart_sharpe(m, path):
    d = m[~m["표본부족"]].dropna(subset=["TO24_Sharpe"]).sort_values("TO24_Sharpe")
    fig, ax = plt.subplots(figsize=(8, max(4, len(d) * 0.22)), dpi=160)
    ax.barh(d["id"] + " " + d["name"].str.slice(0, 24), d["TO24_Sharpe"], color=SERIES[0], height=0.62)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title("2011~2024 초과 샤프(단기채 대비) — 비용·과세 반영, 2025~26 급등장 제외", color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_equity(m, daily, path):
    top = m[~m["표본부족"]].head(5)
    bench = K.load("etf")["069500"]["Close"]
    bench = bench[bench > 0].pct_change()
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    start = pd.Timestamp("2020-01-01")
    for i, (_, r) in enumerate(top.iterrows()):
        eq = (1 + daily[r["id"]].loc[start:].fillna(0)).cumprod()
        ax.plot(eq.index, eq, color=SERIES[i], lw=2, label=f"{r['id']} {r['name'][:18]}")
        ax.text(eq.index[-1], eq.iloc[-1], f" {r['id']}", color=INK2, fontsize=7, va="center")
    b = (1 + bench.loc[start:].fillna(0)).cumprod()
    ax.plot(b.index, b, color=INK2, lw=1.5, ls="--", label="KODEX 200 보유")
    ax.set_title("상위 5개 전략 누적 자산(2020-01=1.0)", color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─── Word ───
META_LINES = [("none", "(c) 국면 없음 [채택]"), ("d3", "d3 점프모형"), ("d1", "d1 가격국면"), ("KODEX200", "(a) KODEX200 보유")]


def chart_meta(md, path):
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    ends = {}
    for (k, lab), col in zip(META_LINES, SERIES):
        eq = (1 + md[k].loc["2012":].fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, color=col, lw=2 if k == "none" else 1.4)
        ends[lab] = eq.iloc[-1]
    y_prev = None
    for lab, y in sorted(ends.items(), key=lambda x: x[1]):   # 라벨 겹침 방지: 로그 간격 최소 13%
        y = y if y_prev is None else max(y, y_prev * 1.13)
        ax.annotate(lab, (md.index[-1], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
        y_prev = y
    ax.set_yscale("log")
    ticks = [1, 1.5, 2, 3, 4, 6]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_title("메타 시스템 누적 성과(2012~, 비용 반영, 로그축, 1 = 시작값)", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlim(pd.Timestamp("2011-10-01"), md.index[-1] + pd.Timedelta(days=900))
    ax.set_xticks(pd.date_range("2012", "2026", freq="2YS"), [str(y) for y in range(2012, 2027, 2)])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


AUDIT_FIXES = [
    ("엔진", "거래정지 중 매매 불가(청산은 정지가 풀린 첫 거래일로 미룸), 상장폐지 종목은 정리매매 마지막 종가에 현금화, "
             "장중 진입은 시가·장중·종가 3시점으로 분리(당일 청산이 다음날 청산되던 문제와 연속 신호일 청산가 오염 수정), "
             "진입일 익절은 max(진입가, 종가)까지만 인정, 손절·익절 기준 = 실제 체결가, 손절 체결에도 슬리피지"),
    ("데이터", "ETF 오류 체결가 2건 교정(114260 2010-02-09, 261240 2019-03-14), DART 월말 스냅샷을 한 보고서 값만 쓰도록 수정"),
    ("PER·PBR", "pykrx 직전 사업연도 연간값(최대 16개월 지연) → DART 기준 시총 ÷ 지배주주 순이익 TTM / 지배주주 자본. "
                "DART 이전(2016-04 전)·누락 종목만 pykrx"),
    ("전략", "SW09 RSI 원문(SMA) 방식, LT22 SUE 12개월 유지, HollyKR 6개 원문 필터·고정% 청산, 단순보유 H 계열 상장 직후 보유, "
             "DF06 상장일 기준, AA12 12개월 이력 전 점수 제외, AA11 DAA 부분배분, LT48 지배주주 ROE, "
             "멀티팩터 결측 항목만 꼴찌 처리, YoY 재무 전략(LT46·LT49·LT50)은 DART 2015 분기 공백 뒤 2018-04 시작"),
    ("설명", "출처와 다른 가정·근사 23건을 전략 설명(assume)에 추가"),
]


BENCH = ["H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09", "BM02"]   # 단순 보유 = 전략이 아니라 비교 기준
EXCL = ["ON05", "CD01", "ON07", "SW14", "SW15", "SW16", "LT14"] + BENCH
PROBLEM = ["AA18", "AA21", "DF01", "DF04", "DF06", "DF07", "LT27", "LT33", "LT50"]   # 최종 독립 검증 퀀트 판정 '문제'
VERDICT = {  # 최종 독립 검증(2026-09-24~25): 출처 등급 A=1차 확인·일치, B=2차만, C=원문과 다름, D=확인 불가 / 퀀트 판정
    "AA01": "A / 주의", "AA02": "B / 건전", "AA03": "D / 주의", "AA05": "B / 건전", "AA06": "C(경미) / 주의",
    "AA08": "D / 주의", "AA09": "C→신호 데이터 수정 / 주의", "AA10": "B / 주의", "AA17": "D / 주의",
    "LT32": "D / 주의(개별종목 중 가장 건전)", "H05": "A / 주의(사후 선택 자산)", "H08": "A / 건전",
}


def final_section(doc, m):
    """최종 결론: 통과 단계, 추천 시스템 vs 비교 기준, 통과 전략."""
    import engine as E
    from datetime import date
    from scipy.stats import norm
    _d = pd.read_parquet(RES / "daily.parquet")
    _n, _T, _g = _d.shape[1], len(_d), 0.5772156649
    _emax = (1 - _g) * norm.ppf(1 - 1 / _n) + _g * norm.ppf(1 - 1 / (_n * np.e))
    _luck = _emax / np.sqrt(_T - 1) * np.sqrt(252)   # N개 무실력 전략에서 우연히 나오는 최대 초과샤프(연율)
    _yrs = _T / 252
    _xt = m.set_index("id").drop(EXCL, errors="ignore")["T24"]
    _nt, _at = int(_xt.notna().sum()), int((_xt > 2).sum())
    _et = _nt * (1 - norm.cdf(2))
    doc.add_heading(f"최종 결론 ({date.today():%Y-%m-%d}, 전수 감사·독립 검증 후)", 1)
    x = m.set_index("id").drop(EXCL, errors="ignore")
    bm = pd.read_csv(RES / "metrics.csv").set_index("id").loc[["H01", "BM02"]]
    bar_all, bar24 = bm["ALL_Sharpe"].max(), bm["TO24_Sharpe"].max()
    steps = [("전체 전략", pd.Series(True, index=x.index)),
             ("전체 기간 연수익률 플러스", x.ALL_CAGR > 0),
             ("앞(2011~19)·뒤(2020~) 구간 모두 플러스", (x.IS_CAGR > 0) & (x.OOS_CAGR > 0)),
             ("비용 2배에서도 뒤 구간 초과 샤프 > 0", x.cost2x_OOS_Sharpe > 0),
             ("초과 샤프 0.5 이상", x.ALL_Sharpe >= 0.5),
             ("퀀트 판정 '문제' 제외", ~x.index.isin(PROBLEM)),
             ("전체·2024년까지 모두 KODEX200·BM02 보다 초과 샤프 높음",
              (x.ALL_Sharpe > bar_all) & (x.TO24_Sharpe > bar24) & (x.index != "H01"))]
    f, rows = pd.Series(True, index=x.index), []
    for nm, c in steps:
        f &= c
        rows.append([nm, int(f.sum())])
    rows[0][1] = len(x) + len(EXCL)
    rows.insert(1, ["수급 결함·체결 불가·벤치마크 제외", len(x)])
    for t in ["결론: 한국 시장에서 이 데이터로 확인된 것은 '종목을 고르는 기술'이 아니라 '자산을 나눠 담아 낙폭을 줄이는 방법'이다. "
              "통과 전략 대부분이 자산배분이고 개별종목 전략은 LT32 하나다.",
              "매달 전략을 고르는 추천 규칙(A)은 KODEX200 45% + 단기채 55% 고정 조합(BM02)에 수익·샤프 모두 졌다. 선택 장치가 가치를 더하지 못했다.",
              f"{_n}개를 시험하면 {_yrs:.0f}년 동안 실력 없이도 초과 샤프 약 {_luck:.2f} 까지 우연히 나온다. "
              f"보정 후(DSR) 95% 를 넘는 것은 체결 불가로 제외한 ON05 하나뿐이고, 그것을 빼면 "
              f"가장 높은 값이 {m.set_index('id').drop(EXCL, errors='ignore')['DSR'].max():.2f} 로 통과 전략이 없다.",
              "국면 판단 4개 정의는 모두 '국면 없음'보다 나쁘다(아래 메타 시스템 장)."]:
        doc.add_paragraph(t, style="List Bullet")
    doc.add_paragraph("판정 기준 구간 변경: 2011~2024 (2025~26 급등장 제외)", style="Heading 3")
    doc.add_paragraph("KODEX200 이 2025년 +94%, 2026년(9월까지) +85% 올라 전체 기간 수치가 급등장에 지배된다. 삼성전자·SK하이닉스 "
                      "비중이 큰 코스피 관련 전략은 이 구간 성과로 유의미성을 검증할 수 없으므로 판정은 2011~2024 로 하고, "
                      "2025~26 은 별도 열로만 보인다. 추가로 KODEX200 에 대한 알파(시장 상승 덕분을 뺀 초과성과)를 본다.")
    xc = m.set_index("id")
    cand = xc[(xc.TO24_Sharpe >= 0.5) & (xc.TO24_CAGR > 0) & (xc.IS_CAGR > 0) & (xc.cost2x_OOS_Sharpe > 0)
              & ~xc.index.isin(PROBLEM + EXCL)].sort_values("TO24_Sharpe", ascending=False)
    rows_c = [[i, r["name"][:36], pct(r.TO24_CAGR), num(r.TO24_Sharpe), pct(r.A24), num(r.T24, 1), num(r.B24),
               pct(r.ALL_MDD), pct(r.R2526), pct(r.DSR, 0),
               "최종 후보" if r.T24 > 2 else "경계(알파 비유의)"] for i, r in cand.iterrows()]
    doc.add_paragraph(f"2011~2024 최종 후보 ({sum(1 for r in rows_c if r[-1] == '최종 후보')}개) + 경계", style="Heading 3")
    table(doc, ["ID", "전략", "11~24 연", "11~24 샤프", "알파(연)", "알파 t", "베타", "전체 MDD", "25~26 연", "DSR", "판정"],
          rows_c, [1.1, 6.8, 1.6, 1.6, 1.5, 1.2, 1.1, 1.6, 1.6, 1.1, 2.6], size=8)
    for t in ["기준: 2011~2024 초과 샤프 0.5 이상·연수익 플러스, 앞 구간 플러스, 비용 2배 생존, 퀀트 판정 '문제' 제외. "
              "알파 t > 2 면 '최종 후보', 아니면 '경계'.",
              "같은 기간 KODEX200: 연 4.7%, 초과 샤프 0.23. 주식45+단기채55(BM02): 초과 샤프 0.16.",
              "단순 보유(H 계열: KODEX200·나스닥100·고배당·국고채)는 매매 규칙이 없고 결과를 보고 고른 자산이라 전략에서 빼고 비교 기준으로만 둔다"
              "(참고: 나스닥100 보유 2011~24 연 16.6% — 미국 기술주·달러 강세의 사후 선택).",
              "주의: AA17·AA09·AA10 의 알파 일부도 미국 주식·달러가 2011~2024 한국보다 좋았던 데서 온 자산 선택 효과다. "
              "순수 한국 전략으로 남은 것은 LT32(연말 배당 교대, 베타 0.19) 하나다.",
              f"{_nt}개를 시험하면 우연만으로도 t > 2 가 {_et:.1f}개 나올 수 있다(실제 {_at}개). "
              f"Harvey et al.(2016) 기준 t > 3 은 0개다. 최종 후보도 '유력'이지 '확정'이 아니며 모의투자로 추가 검증한다."]:
        doc.add_paragraph(t, style="List Bullet")

    doc.add_paragraph("통과 단계 (전체 기간 기준, 참고)", style="Heading 3")
    table(doc, ["기준", "남은 전략 수"], rows, [17, 4], size=9)

    doc.add_paragraph("추천 규칙(A) vs 비교 기준 (2012~2026.09, 비용·과세 반영, 샤프 = 단기채 대비 초과)", style="Heading 3")
    md = pd.read_parquet(RES / "meta_daily.parquet")
    dl = pd.read_parquet(RES / "daily.parquet")
    ser = {"A 추천(매달 TOP5)": md["none"], "KODEX200 보유(H01)": dl["H01"],
           "KODEX200 45% + 단기채 55%(BM02)": dl["BM02"], "나스닥100 보유(H05)": dl["H05"]}
    brow = []
    for k, sr in ser.items():
        sr = sr.dropna().loc["2012":]
        a, b = E.metrics(sr), E.metrics(sr.loc[:"2024"])
        brow.append([k, pct(a["CAGR"]), num(a["Sharpe"]), pct(a["MDD"]), pct(b["CAGR"]), num(b["Sharpe"])])
    table(doc, ["", "연수익률", "초과 샤프", "MDD", "~2024 연수익률", "~2024 초과 샤프"], brow, [7, 2.5, 2.5, 2.5, 3, 3], size=9)

    doc.add_paragraph(f"통과 전략 {int(f.sum())}개", style="Heading 3")
    p = x[f].sort_values("ALL_Sharpe", ascending=False)
    table(doc, ["ID", "전략", "연수익률", "초과 샤프", "MDD", "~2024 연", "~2024 샤프", "DSR", "출처 등급 / 퀀트 판정"],
          [[i, r["name"][:38], pct(r.ALL_CAGR), num(r.ALL_Sharpe), pct(r.ALL_MDD), pct(r.TO24_CAGR), num(r.TO24_Sharpe),
            pct(r.DSR, 0), VERDICT.get(i, "")] for i, r in p.iterrows()],
          [1.2, 7.5, 1.7, 1.6, 1.7, 1.7, 1.7, 1.3, 5.5], size=8)
    doc.add_paragraph("연도별 수익률 (비용·과세 반영)", style="Heading 3")
    ids = list(p.index) + ["H01", "BM02"]
    yr = pd.DataFrame({i: dl[i] for i in ids if i in dl}).assign(**{"A추천": md["none"]})
    yr = (1 + yr.loc["2011":]).groupby(yr.loc["2011":].index.year).prod(min_count=20) - 1
    years = list(yr.index)
    table(doc, ["ID"] + [str(y)[2:] for y in years],
          [[c] + [pct(yr.at[y, c], 0) for y in years] for c in yr.columns],
          [1.4] + [1.35] * len(years), size=7)
    doc.add_paragraph("연도 표기 '26 은 2026-09 까지. 빈칸은 그해 운용 전(상장·데이터 시작 전).")
    for t in ["DSR: 시험한 전략 수만큼의 우연을 보정한 뒤 '진짜 초과 샤프 > 0' 일 확률. 95% 이상이면 우연이 아니라고 말할 수 있다.",
              "~2024: 2025~26 급등장(KODEX200 +94%, +85%)을 뺀 구간. 전체 기간 수치만 인용하지 않는다.",
              "출처 등급 D 는 원 규칙을 1차 자료로 확인하지 못한 것이다. 규칙이 논리적으로 타당해도 '출처대로 구현'이라고 말할 수 없다.",
              "과세(15.4%)는 근사다. 레버리지·인버스·커버드콜 ETF 는 비과세로 가정했다."]:
        doc.add_paragraph(t, style="List Bullet")


def audit_section(doc):
    before = RES / "before_fix" / "metrics.csv"
    if not before.exists():
        return
    a = pd.read_csv(before).set_index("id")
    b = pd.read_csv(RES / "metrics.csv").set_index("id")
    doc.add_heading("전략 전수 감사와 수정 (2026-09-24)", 1)
    doc.add_paragraph("전략 163개와 백테스트 엔진·데이터를 출처 대조, 미래참조, 버그, 설명 불일치 기준으로 전수 감사했다. "
                      "원칙: 버그·미래참조·데이터 오류는 무조건 수정본으로 교체, 출처와 다른 해석은 출처 버전을 공식으로, "
                      "수정 후 성과가 나빠져도 파라미터로 되살리지 않는다. 엔진 수정에는 손계산 대조 테스트를 추가했다(20개 통과).")
    table(doc, ["영역", "수정 내용"], AUDIT_FIXES, [2.5, 23], size=8)
    ex = ["ON05", "CD01", "ON07", "SW14", "SW15", "SW16", "LT14"]
    ok = lambda m: ((m.ALL_Sharpe >= 0.5) & (m.ALL_CAGR > 0) & (m.IS_CAGR > 0) & (m.OOS_CAGR > 0)
                    & (m.cost2x_OOS_Sharpe > 0)).drop(ex, errors="ignore")
    sa, sb = ok(a), ok(b)
    ids = sorted(set(sa[sa].index) | set(sb[sb].index) | set((b.ALL_Sharpe - a.ALL_Sharpe).abs().pipe(lambda x: x[x > 0.15]).index))
    rows = []
    for i in ids:
        if i not in a.index or i not in b.index:
            continue
        st = "유지" if sa.get(i, False) and sb.get(i, False) else "탈락" if sa.get(i, False) else "신규" if sb.get(i, False) else "-"
        rows.append([i, b.loc[i, "name"][:40], num(a.loc[i, "ALL_Sharpe"]), num(b.loc[i, "ALL_Sharpe"]),
                     pct(a.loc[i, "ALL_CAGR"]), pct(b.loc[i, "ALL_CAGR"]), pct(a.loc[i, "ALL_MDD"]), pct(b.loc[i, "ALL_MDD"]), st])
    doc.add_paragraph("수정 전후 비교(전체 기간, 표시 기준 통과 전략 + 샤프가 0.15 넘게 바뀐 전략)", style="Heading 3")
    table(doc, ["ID", "전략", "샤프 전", "샤프 후", "연 전", "연 후", "MDD 전", "MDD 후", "표시 기준"], rows,
          [1.2, 8, 1.5, 1.5, 1.6, 1.6, 1.7, 1.7, 1.8], size=7.5)
    w = RES.parent / "results_warn" / "metrics.csv"
    if w.exists():
        w = pd.read_csv(w).set_index("id")
        ids = [i for i in w.index if i in b.index and i not in ex]
        dsh = (w.loc[ids, "ALL_Sharpe"] - b.loc[ids, "ALL_Sharpe"])
        doc.add_paragraph("부실 전조 필터(참고, 공식 수치에는 미적용)", style="Heading 3")
        doc.add_paragraph(
            "조건: 매수 시점 최근 1년 안에 관리종목·불성실공시·감사의견 공시, 또는 TTM 적자이면서 CB·BW·유상증자 공시. "
            "장기 거래정지 후 부실 폐지된 104건 중 87%가 사전에 걸리고 정상 종목은 14%만 걸린다. "
            f"개별종목 전략 {len(ids)}개에 적용하면 샤프 변화 중앙값 {dsh.median():+.3f}, 개선된 전략 {(dsh > 0).mean():.0%}로 "
            "효과가 작고 필터 조건을 부실 종목을 보고 정해 사후 선택 위험이 있어 공식 수치에는 넣지 않았다. "
            "실매매에서는 보유 종목 경고 표시로 쓴다.")


def meta_section(doc):
    if not (RES / "meta_perf.csv").exists():
        return
    perf = pd.read_csv(RES / "meta_perf.csv")
    tests = pd.read_csv(RES / "meta_tests.csv")
    sig = pd.read_csv(RES / "meta_regime_significance.csv", index_col=0)
    md = pd.read_parquet(RES / "meta_daily.parquet")
    chart_meta(md, RES / "meta.png")
    doc.add_heading("메타 시스템: 국면 판단 → 전략 선택", 1)
    doc.add_paragraph(
        "매월 말 국면을 판정하고, 같은 국면이었던 과거 달(의 다음 달) 성과로 샤프 ≥ 0.5 · 연수익률 > 0 · MDD ≥ -30%를 "
        "통과한 전략만 최대 5개(샤프 순) 골라 슬롯당 20%, 빈 슬롯은 단기채(153130)로 다음 달 운용한다. "
        "같은 국면 표본이 12개월 미만이면 전체 과거로 판정하고, 이력 12개월 미만 전략은 후보에서 뺀다. "
        "(c)는 같은 규칙을 국면 구분 없이 전체 과거로 적용한 버전이다. 규칙·합격 기준은 결과를 보기 전에 고정했다.")
    for s in ["d1 가격: KODEX200 10개월선 × 60일 변동성(확장 중앙값) → 4국면",
              "d2 거시(감마팀 방식): ECOS 성장·물가 z-score, 발표 시차, 히스테리시스 ±0.2 → 4국면",
              "d3 점프모형(jumpmodels, λ=50 고정): KOSPI 지수 특징, 매월 expanding 재적합·online 예측 → 2국면",
              "d4 난기류: KOSPI·원달러·금 마할라노비스 거리(과거 5년), 확장 75% 분위 → 2국면",
              "합격 기준: (c) 대비 샤프 차이가 블록 부트스트랩 + Holm 보정(4개) 후 유의하고, 2012~19와 2020~24 둘 다 (c)보다 높을 것",
              "후보: 전 전략 중 ON05(체결 비현실)와 수급 전략 6개(데이터 결함, 재수집 전)만 제외"]:
        doc.add_paragraph(s, style="List Bullet")
    doc.add_paragraph("국면 유의성 사전 검사", style="Heading 3")
    doc.add_paragraph("차이 = 국면 간 다음 달 수익률 분포가 다른 전략 비율(Kruskal-Wallis p<0.05). 지속성 = 같은 국면이 다시 왔을 때 "
                      "과거 같은 국면의 전략 샤프 순위와 이번 순위의 스피어만 상관 평균. p값은 국면 라벨을 시간축으로 "
                      "순환 이동한 대조군 대비.")
    table(doc, ["국면", "차이 비율", "차이 p", "순위 지속성", "지속성 p"],
          [[k, pct(r.diff_share), num(r.diff_p, 3), num(r.persistence), num(r.persist_p, 3)] for k, r in sig.iterrows()],
          [2, 3, 3, 3, 3], size=9)
    doc.add_paragraph("워크포워드 성과", style="Heading 3")
    names = dict(META_LINES) | {"d2": "d2 거시국면", "d4": "d4 난기류", "EW_all": "(b) 전 전략 동일비중"}
    order = ["none", "d1", "d2", "d3", "d4", "KODEX200", "EW_all"]
    per = ["2012-2019", "2020-2024", "2025-2026", "ALL"]
    rows = []
    for k in order:
        g = perf[perf["mode"] == k].set_index("period")
        rows.append([names[k]] + [f"{num(g.loc[p, 'Sharpe'])} / {pct(g.loc[p, 'CAGR'])}" for p in per] + [pct(g.loc["ALL", "MDD"])])
    table(doc, ["방식", "2012~19 샤프/CAGR", "2020~24", "2025~26", "전체", "전체 MDD"], rows, [5, 3.5, 3.5, 3.5, 3.5, 2.5], size=9)
    doc.add_picture(str(RES / "meta.png"), width=Cm(22))
    doc.add_paragraph("합격 판정((c) 대비)", style="Heading 3")
    table(doc, ["국면", "샤프 차이 2012~24", "p", "Holm p", "2012~19 차이", "2020~24 차이", "판정"],
          [[r["mode"], num(r.sharpe_diff_12_24), num(r.p, 3), num(r.p_holm, 3), num(r.diff_12_19), num(r.diff_20_24),
            "합격" if r["pass"] else "불합격"] for _, r in tests.iterrows()], [2, 3, 2, 2, 3, 3, 2], size=9)
    log = pd.read_csv(RES / "meta_log_none.csv")
    from collections import Counter
    cnt = Counter(x for p in log.picks.dropna() for x in p.split(","))
    doc.add_paragraph("결론", style="Heading 3")
    for s in ["4개 국면 정의 모두 불합격 → 사전 규칙대로 국면 판단을 버리고 (c)를 채택한다.",
              "전략 순위는 국면과 무관하게 지속된다(지속성 상관은 높지만 무작위 라벨과 차이 없음). "
              "'특정 국면에 강한 전략'보다 '원래 좋은 전략이 계속 좋은' 구조다.",
              f"(c)의 선택은 저변동 자산 위주다(월 선택 빈도 상위: "
              + ", ".join(f"{k} {v}회" for k, v in cnt.most_common(6)) + f", 총 {len(log)}개월). "
              "샤프 기준이라 채권 보유(H09)가 자주 뽑히며, 연수익률은 KODEX200보다 낮고 변동·낙폭이 작다.",
              f"최근 결정({log.decision.iloc[-1]} 말): " + (log.picks.iloc[-1] or "전량 단기채") + " 각 20%."]:
        doc.add_paragraph(s, style="List Bullet")


def set_font(doc):
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(9)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)


def shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_)
    tcPr.append(shd)


def table(doc, header, rows, widths=None, size=7.5):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = h
        shade(c, "DCE6F2")
        c.paragraphs[0].runs[0].font.bold = True
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)
    for row in t.rows:
        for i, c in enumerate(row.cells):
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
            if widths:
                c.width = Cm(widths[i])
    return t


COLS = ["순위", "ID", "전략", "분류", "11~24 연", "11~24 샤프", "알파(연)", "알파 t", "베타", "전체 MDD",
        "25~26 연", "전체 연", "승률", "손익비", "연회전율", "DSR", "검증"]
W = [0.8, 1.0, 5.0, 1.4, 1.3, 1.2, 1.2, 1.0, 0.9, 1.3, 1.3, 1.3, 1.0, 1.0, 1.2, 1.0, 2.2]


def rowvals(r):
    return [("" if pd.isna(r["순위"]) else int(r["순위"])), r["id"], r["name"] + (" [ETF]" if r["etf"] else ""),
            r["cat"], pct(r.get("TO24_CAGR")), num(r.get("TO24_Sharpe")), pct(r.get("A24")), num(r.get("T24"), 1),
            num(r.get("B24")), pct(r["ALL_MDD"]), pct(r.get("R2526")), pct(r["ALL_CAGR"]),
            pct(r["ALL_WinRate"], 0), num(r["ALL_Payoff"]), num(r["Turnover"], 1), pct(r.get("DSR"), 0), flag(r)]


def build():
    m = load()
    daily = pd.read_parquet(RES / "daily.parquet")
    chart_sharpe(m, RES / "sharpe.png")
    chart_equity(m, daily, RES / "equity.png")
    m.to_csv(RES / "ranking.csv", index=False, encoding="utf-8-sig")

    doc = Document()
    set_font(doc)
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(1.5))

    doc.add_heading("한국시장 매매전략 백테스트 순위", 0)
    n_ok = (~m["표본부족"]).sum()
    doc.add_paragraph(
        f"전략 {len(m)}개를 같은 데이터·비용·체결 규칙으로 백테스트했다(순위 대상 {n_ok}개, 표본부족 {len(m) - n_ok}개). "
        "엔진은 오픈소스 vectorbt 1.1이며 수치는 모두 이 코드로 계산한 값이다.")
    doc.add_heading("조건", 1)
    for s in [
        "데이터: KRX 일별 전종목(KOSPI·KOSDAQ, 상장폐지 포함, 2010-01~2026-09), 수정주가는 KRX 등락률로 복원. ETF는 KRX OpenAPI ETF 일별(휴장일 빈 행 제거, 등락률로 수정). 재무는 DART 일괄 재무제표(결산+90일/분기+45일 사용), 공시 이벤트는 OpenDART 목록(2015~).",
        "유니버스: 보통주 + 신호일 시가총액 1,000억 원 이상(우선주·스팩·리츠·외국주권 제외).",
        "비용: 수수료 편도 0.015% + 매도 거래세(연도별 0.30%→0.15~0.20%, ETF 0) + 슬리피지 편도(시총 5조↑ 0.1%, 1조↑ 0.2%, 그 외 0.5%, ETF 0.05%).",
        "체결: 신호 다음 날 시가 진입(종가형은 당일 종가), 다음 날 시가 +29% 이상 출발 시 매수 불가, 거래정지일 무체결, 상장폐지는 마지막 종가 청산.",
        "포트폴리오: 신호형은 최대 10종목 동일비중(초과 시 전일 거래대금 순), 리밸런싱형은 목표비중 다음 날 시가 체결.",
        "기간: 표본내(IS) 2011~2019, 표본외(OOS) 2020~2026.09. 수급 전략(CD01·ON07·SW14·SW15·SW16·LT14)은 수급 데이터가 생존편향·수집 차단으로 불완전해 결과를 신뢰하지 않는다(재수집 전).",
        f"순위: OOS 거래 {MIN_OOS_TRADES}건 이상만. 종합점수 = OOS 샤프·칼마·PF·거래당 기대수익의 백분위 평균.",
        "리밸런싱형(중장기·자산배분)의 승률·손익비·PF·거래수는 월간 수익률 기준(거래수=개월수).",
        "샤프·소르티노는 단기채(KODEX 단기채권, 상장 전은 국고채3년) 대비 초과수익 기준이다(2026-09-25 변경, 이전 보고서는 무위험수익률 0).",
        "과세: 국내상장 채권·금·달러·원유·해외·리츠 ETF 매매차익 배당소득세 15.4%를 연말에 그해 과세 자산 기여 이익 기준으로 근사 차감(미실현 이익 포함·연내 상계 → 근사). 레버리지·인버스·커버드콜은 국내주식형 비과세로 가정.",
        "~2024 CAGR: 2025~26 급등장을 뺀 2011~2024 연수익률. DSR(Deflated Sharpe): 시험한 전략 수만큼의 우연 샤프를 보정한 뒤 '진짜 샤프 > 0' 확률(95% 이상이면 우연 아님).",
        "벤치마크: H01(KODEX200 보유), BM02(KODEX200 45% + 단기채 55% 월간 고정).",
        "검증 표시: 견고=IS·OOS 모두 플러스 / 비용OK=비용 2배에서도 OOS 샤프>0 / 랜덤↑=같은 날 같은 수를 무작위 종목으로 샀을 때보다 OOS 샤프가 높음.",
    ]:
        doc.add_paragraph(s, style="List Bullet")

    final_section(doc, m)

    doc.add_heading("비교 기준 (단순 보유·고정 비중 — 전략 아님)", 1)
    bm_ = m[m["id"].isin(BENCH)]
    table(doc, COLS, [rowvals(r) for _, r in bm_.iterrows()], W)
    m = m[~m["id"].isin(BENCH)].copy()
    m.loc[~m["표본부족"], "순위"] = np.arange(1, (~m["표본부족"]).sum() + 1)

    doc.add_heading("종합 순위 (2011~2024 초과 샤프 순)", 1)
    doc.add_paragraph("기간: '11~24' = 2011~2024(판정 기준 구간, 2025~26 급등장 제외), '25~26' = 2025-01~2026-09 연율, "
                      "'전체' = 전략 시작~2026-09. 알파·베타는 2011~2024 KODEX200 초과수익 회귀(알파 t > 2 면 통계적으로 의미). "
                      "승률·손익비는 전체 기간(리밸런싱형은 월간 기준).")
    table(doc, COLS, [rowvals(r) for _, r in m[~m["표본부족"]].iterrows()], W)
    doc.add_picture(str(RES / "equity.png"), width=Cm(22))
    doc.add_picture(str(RES / "sharpe.png"), width=Cm(18))

    doc.add_heading("지표별 상위 10", 1)
    ok = m[~m["표본부족"]]
    for title, col, asc, fmt in [("2011~2024 초과 샤프", "TO24_Sharpe", False, num), ("OOS(2020~) MDD(낮은 순)", "OOS_MDD", False, pct),
                                 ("OOS 승률", "OOS_WinRate", False, pct), ("OOS 손익비", "OOS_Payoff", False, num),
                                 ("OOS Profit Factor", "OOS_PF", False, num), ("연 회전율(낮은 순)", "Turnover", True, num)]:
        doc.add_paragraph(title, style="Heading 3")
        top = ok.sort_values(col, ascending=asc).head(10)
        table(doc, ["ID", "전략", col, "11~24 샤프", "11~24 연"],
              [[r["id"], r["name"], fmt(r[col]), num(r["TO24_Sharpe"]), pct(r["TO24_CAGR"])] for _, r in top.iterrows()],
              [1.2, 8, 2.5, 2, 2])

    sens = sensitivity(m)
    if sens is not None:
        doc.add_heading("슬리피지 민감도", 1)
        doc.add_paragraph("같은 전략을 슬리피지만 바꿔 다시 돌린 결과(OOS 2020~). 기존 = 5조↑0.1%/1조↑0.2%/그 외 0.5%, "
                          "낮춤 = 0.05%/0.1%/0.2%, 없음 = 수수료·거래세만. ETF 슬리피지(0.05%)는 모든 시나리오 동일.")
        table(doc, ["ID", "전략", "CAGR 기존", "CAGR 낮춤", "CAGR 없음", "샤프 기존", "샤프 낮춤", "샤프 없음",
                    "IS CAGR 낮춤", "연회전율"],
              [[r["id"], r["name"], pct(r["OOS_CAGR"]), pct(r["low_OOS_CAGR"]), pct(r["zero_OOS_CAGR"]),
                num(r["OOS_Sharpe"]), num(r["low_OOS_Sharpe"]), num(r["zero_OOS_Sharpe"]),
                pct(r["low_IS_CAGR"]), num(r["Turnover"], 1)] for _, r in sens.iterrows()],
              [1.2, 7.5, 1.7, 1.7, 1.7, 1.5, 1.5, 1.5, 1.8, 1.5])

    audit_section(doc)
    meta_section(doc)

    doc.add_heading("분류별 결과", 1)
    for cat, g in m.groupby("cat", sort=False):
        doc.add_paragraph(cat, style="Heading 3")
        table(doc, COLS, [rowvals(r) for _, r in g.iterrows()], W)

    if m["표본부족"].any():
        doc.add_heading("순위 제외: 표본 부족 또는 체결 비현실", 1)
        for k, v in UNREALISTIC.items():
            doc.add_paragraph(f"{k}: {v}", style="List Bullet")
        table(doc, COLS, [rowvals(r) for _, r in m[m["표본부족"]].iterrows()], W)

    doc.add_heading("전략 정의·출처·가정", 1)
    table(doc, ["ID", "전략", "출처", "구현 가정"],
          [[r["id"], r["name"], r["src"], "" if pd.isna(r["assume"]) else r["assume"]] for _, r in
           m.sort_values("id").iterrows()], [1.2, 8, 7, 9])

    doc.add_heading("백테스트하지 않은 전략", 1)
    table(doc, ["전략", "사유"], EXCLUDED, [12, 13])

    doc.add_heading("한계", 1)
    for s in [
        "슬롯 포트폴리오는 보유 종목을 매일 1/N로 재조정한다고 가정했다(실제는 진입 시점 비중이 유지됨).",
        "하한가 매도 불가·호가 잔량 부족은 반영하지 않았다. 상한가 종가 매수(ON05)는 실제 체결이 어려워 결과가 과대평가됐을 수 있다.",
        "조건식 중 청산 규칙이 없는 것은 공통 청산(익절 10%/손절 3%/5일)을 붙였으므로 원작자 성과와 다를 수 있다.",
        "메타 시스템의 전략 파라미터는 출처 저자들이 과거 데이터를 보고 정한 것이라 2011년 관점의 완전한 표본외가 아니다. 국면당 표본이 3~4년이라 추정 오차가 크다.",
        "잠정실적 서프라이즈 전략은 공시 목록에 실적 수치가 없어 구현하지 않았다.",
        "PER·PBR은 DART(시총 ÷ 지배주주 TTM 순이익·자본, 법정 제출기한 후 사용)로 계산했고 2016-04 이전·DART 누락 종목만 pykrx 연간값이다. 배당·EPS는 pykrx 연간값이다. 시총은 보통주 기준이라 우선주가 있는 기업의 PER은 약간 낮게 나온다.",
        "정지 종목 매도가 미뤄지는 동안 다른 종목 매수는 남은 현금만큼만 부분 체결된다. PER 분모 TTM 순이익에 일회성 이익이 포함된다.",
        "상장폐지 중 정리매매 가격이 없고 60거래일 이상 정지된 경우는 정리매매 197건 중앙값(정지 직전가 대비 -94.2%)으로 청산했다.",
        "DART 일괄 원천에 2015 사업연도 1~3분기 보고서가 없어, 12개월 전 재무와 비교하는 전략은 2018-04부터 시작한다.",
        "과거 성과는 미래 수익을 보장하지 않는다. 상위 전략은 모의투자로 추가 검증한 뒤 사용한다.",
    ]:
        doc.add_paragraph(s, style="List Bullet")
    doc.save(OUT)
    print(f"저장: {OUT}")
    return m


if __name__ == "__main__":
    m = build()
    cols = ["순위", "id", "name", "OOS_CAGR", "OOS_Sharpe", "OOS_MDD", "OOS_WinRate", "OOS_Payoff", "Turnover"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 100):
        print(m[cols].to_string())
