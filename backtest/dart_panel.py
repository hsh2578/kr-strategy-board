"""DART 일괄다운로드(재무상태표·손익·현금흐름) → 월말 시점 재무 패널.

- 연결 우선, 없으면 별도. 12월 결산 법인만.
- 손익·현금흐름은 누적(YTD) 값으로 TTM(최근 4분기 합) 계산:
    TTM(y,q) = YTD(y,q) + FY(y-1) - YTD(y-1,q),  q=FY 면 FY(y)
- 사용 가능일 = 결산기준일 + 90일(사업보고서) / +45일(분기·반기) — 법정 제출기한. 그 전엔 값을 쓰지 않는다.
출력: data/panel/fin_<항목>.parquet (월말 × 종목)

    python dart_panel.py
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
BULK = ROOT / "data" / "dart_bulk"
PANEL = ROOT / "data" / "panel"

ITEMS = {  # 항목코드(접두어 제거) → 이름
    "Revenue": "rev", "GrossProfit": "gp", "OperatingIncomeLoss": "op", "ProfitLoss": "ni",
    "CashFlowsFromUsedInOperatingActivities": "ocf",
    "Assets": "assets", "Liabilities": "liab", "Equity": "equity",
    "CurrentAssets": "cur_assets", "CurrentLiabilities": "cur_liab",
    # PER·PBR 용 지배주주 몫(연결만 존재, 별도 재무제표 기업은 ni·equity 로 대체)
    "ProfitLossAttributableToOwnersOfParent": "ni_own", "EquityAttributableToOwnersOfParent": "equity_own",
    "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": "capex",   # 잉여현금흐름 = 영업CF - |유형자산 취득|
}
FLOW = {"rev", "gp", "op", "ni", "ocf", "ni_own", "capex"}
REPORT_Q = {"1분기보고서": 1, "반기보고서": 2, "3분기보고서": 3, "사업보고서": 4}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def read_zip(path: Path) -> pd.DataFrame:
    out = []
    z = zipfile.ZipFile(path)
    for n in z.namelist():
        txt = z.read(n).decode("cp949", errors="replace")
        df = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        # 값 열: 헤더 12번째(0-based) = '당기…' 첫 열(분기 손익은 3개월/누적 중 누적을 쓴다)
        vals = [c for c in df.columns[12:] if c.startswith("당기")]
        if not vals:
            continue
        cum = [c for c in vals if "누적" in c]
        vcol = cum[0] if cum else vals[0]
        code = df["항목코드"].str.replace(r"^(ifrs-full|ifrs|dart)_", "", regex=True)
        keep = code.isin(ITEMS)
        if not keep.any():
            continue
        d = pd.DataFrame({
            "code": df.loc[keep, "종목코드"].str.strip("[] "),
            "cons": df.loc[keep, "재무제표종류"].str.contains("연결"),
            "fye": df.loc[keep, "결산월"].str.strip(),
            "end": pd.to_datetime(df.loc[keep, "결산기준일"].str.strip(), errors="coerce"),
            "rep": df.loc[keep, "보고서종류"].str.strip(),
            "item": code[keep].map(ITEMS),
            "val": _num(df.loc[keep, vcol]),
        })
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build():
    raw = pd.concat([read_zip(p) for p in sorted(BULK.glob("*.zip"))], ignore_index=True)
    raw = raw[(raw["fye"] == "12") & raw["val"].notna() & raw["rep"].isin(REPORT_Q)]
    raw["q"] = raw["rep"].map(REPORT_Q)
    raw["y"] = raw["end"].dt.year
    # 연결 우선: 같은 (종목,연,분기,항목) 에 연결이 있으면 연결만
    raw = raw.sort_values("cons", ascending=False).drop_duplicates(["code", "y", "q", "item"])
    wide = raw.pivot_table(index=["code", "y", "q"], columns="item", values="val", aggfunc="first")
    ends = raw.groupby(["code", "y", "q"])["end"].first()
    # 단위 오류(원·천원·백만원 혼동) 보고서 제외: 자산총계가 같은 회사 앞뒤 보고서 중앙값 대비 100배 넘게 튀면 버린다
    a = wide["assets"].sort_index()
    ref = a.groupby(level="code").transform(lambda s: pd.concat([s.shift(1), s.shift(-1)], axis=1).median(axis=1))
    bad = ((a / ref) > 100) | ((a / ref) < 0.01)
    print(f"단위 오류 의심 보고서 제외: {int(bad.sum())}건")
    wide, ends = wide.drop(index=bad[bad].index), ends.drop(index=bad[bad].index)

    # TTM
    flows = [c for c in wide.columns if c in FLOW]
    ytd = wide[flows]
    fy = ytd.xs(4, level="q")
    ttm = ytd.copy()
    for (code, y, q), row in ytd.iterrows():
        if q == 4:
            continue
        try:
            ttm.loc[(code, y, q)] = row + fy.loc[(code, y - 1)] - ytd.loc[(code, y - 1, q)]
        except KeyError:
            ttm.loc[(code, y, q)] = np.nan
    fin = pd.concat([ttm.add_suffix("_ttm"), wide.drop(columns=flows)], axis=1)
    fin["end"] = ends
    fin["avail"] = fin["end"] + pd.to_timedelta(np.where(fin.index.get_level_values("q") == 4, 90, 45), unit="D")
    fin = fin.reset_index()

    # 월말 패널: 그 월말까지 사용 가능해진 가장 최근 보고서
    close = pd.read_parquet(PANEL / "close.parquet")
    me = pd.DatetimeIndex(close.index.to_series().groupby(close.index.to_period("M")).last())
    fin = fin.sort_values("avail")
    cols = [c for c in fin.columns if c not in ("code", "y", "q", "end", "avail")]
    snaps = []
    for t in me:
        s = fin[fin["avail"] <= t].drop_duplicates("code", keep="last").set_index("code")  # 한 보고서 값만(항목 섞임 방지)
        s["date"] = t
        snaps.append(s.reset_index())
    snap = pd.concat(snaps, ignore_index=True)
    for c in cols + ["end"]:
        w = snap.pivot(index="date", columns="code", values=c).reindex(columns=close.columns)
        w.to_parquet(PANEL / f"fin_{c}.parquet")
    print(f"재무 원천 {len(raw):,}행 · 종목 {raw['code'].nunique()} · 기간 {raw['y'].min()}~{raw['y'].max()} · 항목 {cols}")


if __name__ == "__main__":
    build()
