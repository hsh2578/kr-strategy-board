"""백테스트 입력 패널 구축·로드.

- 개별종목: KRX 일별 전종목 원천(상장폐지 포함) → 날짜×종목 wide 패널, 수정주가 복원
- ETF·지수: FinanceDataReader
- 월말 PER/PBR/DIV: pykrx (KRX 로그인)
- 수급: 급등주 분석프로그램 investor.pkl (2021~)

    python kr_data.py build        # 패널 생성 (collect.py 이후)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
PANEL = DATA / "panel"
MIN_CAP = 1e11  # 사용자 지정: 시총 1,000억 이상
FIELDS = ["open", "high", "low", "close", "ret", "volume", "trdval", "mktcap"]
ETFS = {
    "069500": "KODEX 200", "229200": "KODEX 코스닥150", "122630": "KODEX 레버리지",
    "233740": "KODEX 코스닥150레버리지", "148070": "KOSEF 국고채10년", "114260": "KODEX 국고채3년",
    "132030": "KODEX 골드선물(H)", "261240": "KODEX 미국달러선물", "360750": "TIGER 미국S&P500",
    "379800": "KODEX 미국S&P500TR", "153130": "KODEX 단기채권", "143850": "TIGER 미국S&P500선물(H)",
    "305080": "TIGER 미국채10년선물", "133690": "TIGER 미국나스닥100",
    # 2차: 자산배분·섹터 로테이션용
    "195970": "PLUS 선진국MSCI(H)", "195980": "PLUS 신흥국MSCI(H)", "182480": "TIGER 미국MSCI리츠(H)",
    "130680": "TIGER 원유선물(H)", "280930": "KODEX 미국러셀2000(H)", "304660": "KODEX 미국30년국채울트라(H)",
    "329750": "TIGER 미국달러단기채권", "226980": "KODEX 200중소형", "147970": "TIGER 모멘텀",
    "167860": "KIWOOM 국고채10년레버리지", "138230": "KIWOOM 미국달러선물", "136340": "RISE 중기우량회사채",
    "101280": "KODEX 일본TOPIX100", "195930": "TIGER 유로스탁스50(H)", "192090": "TIGER 차이나CSI300",
    "225040": "TIGER 미국S&P500레버리지(H)", "152380": "KODEX 국채선물10년", "273130": "KODEX 종합채권액티브",
    "139220": "TIGER 200 건설", "139230": "TIGER 200 중공업", "139240": "TIGER 200 철강소재",
    "139250": "TIGER 200 에너지화학", "139260": "TIGER 200 IT", "139270": "TIGER 200 금융",
    "139280": "TIGER 200 경기소비재", "139290": "TIGER 200 경기방어", "143860": "TIGER 헬스케어",
    "227550": "TIGER 200 산업재", "227560": "TIGER 200 생활소비재",
    # 3차: 방어·위기 국면용
    "114800": "KODEX 인버스", "266410": "KODEX 필수소비재", "227540": "TIGER 200 헬스케어",
    "279540": "KODEX 최소변동성", "261220": "KODEX WTI원유선물(H)", "289480": "TIGER 200커버드콜",
    "161510": "PLUS 고배당주",
}
SECTOR_ETFS = ["139220", "139230", "139240", "139250", "139260", "139270", "139280", "139290", "143860"]
US = ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY", "TIP", "BND", "VWO", "BIL", "TLT"]
INVESTOR_PKL = Path(r"C:/Users/hsh/Desktop/급등주 분석프로그램/data/investor.pkl")


def _common_mask(codes: pd.Series, names: pd.Series) -> pd.Series:
    """보통주만: 단축코드 끝자리 0(우선주 5/7/9/K 제외), 외국주권(9xxxxx)·스팩·리츠·인프라 제외."""
    nm = names.astype(str)
    keep = codes.str.len().eq(6) & codes.str[-1].eq("0") & ~codes.str.startswith("9")
    keep &= ~nm.str.contains(r"스팩|리츠|인프라|제\d+호", regex=True, na=False)
    return keep


def build_stocks():
    raw = []
    for m in ("kospi", "kosdaq"):
        df = pd.read_parquet(DATA / f"krx_raw_{m}.parquet")
        raw.append(df)
    df = pd.concat(raw, ignore_index=True).rename(columns={
        "BAS_DD": "date", "ISU_CD": "code", "ISU_NM": "name", "MKT_NM": "market",
        "TDD_OPNPRC": "open", "TDD_HGPRC": "high", "TDD_LWPRC": "low", "TDD_CLSPRC": "close",
        "FLUC_RT": "chg_pct", "ACC_TRDVOL": "volume", "ACC_TRDVAL": "trdval", "MKTCAP": "mktcap"})
    df = df[_common_mask(df["code"], df["name"])]
    df["ret"] = df["chg_pct"] / 100.0
    # 한 번이라도 시총 1,000억 이상이었던 종목만 남겨 패널 크기를 줄인다
    ever = df.groupby("code")["mktcap"].max()
    df = df[df["code"].isin(ever[ever >= MIN_CAP].index)]

    PANEL.mkdir(parents=True, exist_ok=True)
    wide = {f: df.pivot(index="date", columns="code", values=f).astype("float64") for f in FIELDS}
    halted = (wide["volume"].fillna(0) <= 0) | (wide["open"].fillna(0) <= 0)
    for f in ("open", "high", "low"):
        wide[f] = wide[f].mask(halted)
    wide["close"] = wide["close"].where(wide["close"] > 0)
    # 수정주가: KRX 등락률(권리락·분할 조정 기준가 대비)을 누적해 마지막 종가에 맞춘다
    cum = (1 + wide["ret"].fillna(0)).cumprod().where(wide["close"].notna())
    last_close = wide["close"].ffill().iloc[-1]
    last_cum = cum.ffill().iloc[-1]
    factor = (cum * (last_close / last_cum)) / wide["close"]
    for f in ("open", "high", "low", "close"):
        wide[f"adj_{f}"] = wide[f] * factor
    for k, v in wide.items():
        v.to_parquet(PANEL / f"{k}.parquet")
    meta = (df.sort_values("date").groupby("code")
              .agg(name=("name", "last"), market=("market", "last"),
                   first=("date", "first"), last=("date", "last")))
    meta.to_parquet(PANEL / "meta.parquet")
    print(f"종목 {len(meta)} · 기간 {wide['close'].index[0].date()}~{wide['close'].index[-1].date()}")


# ETF 오류 체결가(종가만 튀고 다음날 원복, 2026-09-24 감사에서 FDR·시가와 대조해 확인)
BAD_TICKS = [("114260", "2010-02-09"), ("261240", "2019-03-14")]


# 매매차익 과세(15.4%) 대상이 아닌 국내 주식형 ETF. 나머지(채권·금·달러·원유·해외·리츠)는 과세.
# 레버리지·인버스·커버드콜은 국내주식 기반으로 보고 비과세 처리(확인 필요, 보고서에 명시).
DOMESTIC_EQUITY_ETFS = {"069500", "229200", "122630", "233740", "226980", "147970", "114800", "289480", "161510",
                        "266410", "227540", "279540", "143860", "227550", "227560",
                        "139220", "139230", "139240", "139250", "139260", "139270", "139280", "139290"}


def build_prefcap():
    """보통주별 우선주 시총 합(같은 앞 5자리 코드). PER·PBR 분자 = 보통주 + 우선주 시총."""
    raw = pd.concat([pd.read_parquet(DATA / f"krx_raw_{m}.parquet") for m in ("kospi", "kosdaq")], ignore_index=True)
    c = raw["ISU_CD"].astype(str)
    pref = raw[c.str.len().eq(6) & ~c.str[-1].eq("0") & ~c.str.startswith("9")]
    pref = pref.assign(common=pref["ISU_CD"].str[:5] + "0")
    w = pref.pivot_table(index="BAS_DD", columns="common", values="MKTCAP", aggfunc="sum").astype("float64")
    w.index = pd.to_datetime(w.index)
    close = load("close")
    w = w.reindex(index=close.index, columns=close.columns)
    w.to_parquet(PANEL / "pref_cap.parquet")
    print(f"우선주 시총: 보통주 {int(w.notna().any().sum())}곳")


def build_etfs():
    """KRX ETF 일별 원천(2010~)으로 구축. FDR(네이버)은 최근 3,000일만 줘서 쓰지 않는다.
    가격은 등락률(분배락·분할 조정 기준가 대비)로 수정주가를 복원한다."""
    raw = pd.read_parquet(DATA / "krx_raw_etf.parquet")
    raw = raw[raw["ISU_CD"].isin(ETFS) & raw["TDD_CLSPRC"].notna()]   # 휴장일에도 빈 행이 와서 제거
    f = {k: raw.pivot(index="BAS_DD", columns="ISU_CD", values=v).astype("float64")
         for k, v in (("Open", "TDD_OPNPRC"), ("High", "TDD_HGPRC"), ("Low", "TDD_LWPRC"),
                      ("Close", "TDD_CLSPRC"), ("Volume", "ACC_TRDVOL"), ("ret", "FLUC_RT"))}
    for code, day in BAD_TICKS:   # 확인된 오류 체결가: 종가를 시가로, 그날·다음날 등락률 재계산
        d = pd.Timestamp(day)
        f["Close"].at[d, code] = f["High"].at[d, code] = f["Open"].at[d, code]
        cl = f["Close"][code].dropna()
        i = cl.index.get_loc(d)
        for j in (i, i + 1):
            f["ret"].at[cl.index[j], code] = (cl.iloc[j] / cl.iloc[j - 1] - 1) * 100
    close = f["Close"].where(f["Close"] > 0)
    cum = (1 + (f["ret"] / 100).fillna(0)).cumprod().where(close.notna())
    factor = cum * (close.ffill().iloc[-1] / cum.ffill().iloc[-1]) / close
    out = {}
    for code in ETFS:
        if code not in close:
            continue
        ok = f["Volume"][code].fillna(0) > 0
        out[code] = pd.DataFrame({k: (f[k][code] * factor[code]).where(ok | (k == "Close"))
                                  for k in ("Open", "High", "Low", "Close")} | {"Volume": f["Volume"][code]})
    etf = pd.concat(out, axis=1)
    etf.index.name = "Date"
    etf.to_parquet(PANEL / "etf.parquet")
    print(f"ETF {len(out)}개 · {etf.index[0].date()}~{etf.index[-1].date()}")


def build_us():
    """Keller 계열 신호 계산용 미국 ETF 종가 + 미국 실업률(FRED). 매매는 국내 ETF로 한다."""
    import FinanceDataReader as fdr
    c = pd.DataFrame({u: fdr.DataReader(u, "2008-01-01")["Adj Close"] for u in US})   # 분배금 반영 총수익(Keller 규칙)
    c.to_parquet(PANEL / "us.parquet")
    fdr.DataReader("FRED:UNRATE", "2008-01-01").to_parquet(PANEL / "unrate.parquet")
    print(f"미국 {len(US)}개 + 실업률")


def build_longhist():
    """국면 모델 학습용 장기 시계열(Yahoo 경유 FDR): KOSPI 1997~, 원달러 2003~, 금 2000~."""
    import FinanceDataReader as fdr
    c = pd.DataFrame({"kospi": fdr.DataReader("^KS11", "1997-01-01")["Close"],
                      "usdkrw": fdr.DataReader("USD/KRW", "2003-01-01")["Close"],
                      "gold": fdr.DataReader("GC=F", "2000-01-01")["Close"]})
    c.to_parquet(PANEL / "longhist.parquet")
    print({k: str(v.dropna().index[0].date()) for k, v in c.items()})


def build_fundamentals():
    """월말 PER/PBR/DIV 스냅샷 (전 종목)."""
    sys.path.insert(0, r"C:/Users/hsh/Desktop")
    from env_loader import load_env
    load_env()
    from pykrx import stock
    idx = pd.DatetimeIndex(pd.read_parquet(DATA / "krx_raw_kospi.parquet", columns=["BAS_DD"])["BAS_DD"].unique()).sort_values()
    month_ends = idx.to_series().groupby(idx.to_period("M")).last()
    path = PANEL / "fundamental.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    done = set(have["date"]) if len(have) else set()
    rows = [have] if len(have) else []
    for d in month_ends:
        if d in done:
            continue
        f = stock.get_market_fundamental(d.strftime("%Y%m%d"), market="ALL")
        if f is None or f.empty:
            continue
        f = f.reset_index().rename(columns={"티커": "code"})
        f["date"] = d
        rows.append(f)
        if len(rows) % 20 == 0:
            pd.concat(rows, ignore_index=True).to_parquet(path)
    pd.concat(rows, ignore_index=True).to_parquet(path)
    print(f"펀더멘털 월 {len(month_ends)}개")


def build_investor():
    """collect_flow.py 산출(pykrx 날짜별 전종목, 상장폐지 포함, 백만원)로 구축."""
    raw = pd.read_parquet(DATA / "flow_raw.parquet")
    for f in ("foreign", "inst"):
        w = raw[raw["field"] == f].pivot_table(index="date", columns="code", values="net", aggfunc="sum")
        w.to_parquet(PANEL / f"flow_{f}.parquet")
    print(f"수급 {raw['code'].nunique()}종목 · {raw['date'].min().date()}~{raw['date'].max().date()}")


def build_investor_old():
    d = pd.read_pickle(INVESTOR_PKL)  # 사용자 본인 프로젝트가 만든 로컬 캐시(신뢰 소스) — 절반·현존종목뿐이라 폐기
    frames = {c: v.set_index("date")[["foreign", "inst"]] for c, v in d.items()}
    df = pd.concat(frames, axis=1)
    df.columns.names = ["code", "field"]
    for f in ("foreign", "inst"):
        df.xs(f, axis=1, level="field").to_parquet(PANEL / f"flow_{f}.parquet")
    print(f"수급 {len(frames)}종목")


# ─── 로드 ───
_cache = {}


def load(name: str) -> pd.DataFrame:
    if name not in _cache:
        _cache[name] = pd.read_parquet(PANEL / f"{name}.parquet")
    return _cache[name]


def universe() -> pd.DataFrame:
    """t일 종가 기준 시총 1,000억 이상(신호일 t 에 사용 → t+1 체결이므로 '전일 시총' 기준)."""
    return load("mktcap") >= MIN_CAP


def etf(code: str) -> pd.DataFrame:
    return load("etf")[code].dropna(how="all")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "build"
    PANEL.mkdir(parents=True, exist_ok=True)
    if step == "build":
        build_stocks(); build_etfs(); build_investor(); build_fundamentals()
    else:
        globals()[f"build_{step}"]()
