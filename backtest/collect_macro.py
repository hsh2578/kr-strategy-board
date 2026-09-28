"""한국은행 ECOS 거시지표 6개 수집(감마팀 거시 국면 재현용). 시스템/venv 어느 파이썬이든 가능.

통계표·항목 코드는 ECOS StatisticTableList/StatisticItemList 로 확인한 값.
출력: data/panel/macro_raw.parquet (월말 인덱스, 원 수준값; GDP는 분기말)

    python collect_macro.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, r"C:/Users/hsh/Desktop")
from env_loader import load_env  # noqa: E402

load_env()
KEY = os.environ["BOK_ECOS_API_KEY"]
OUT = Path(__file__).parent / "data" / "panel" / "macro_raw.parquet"

SERIES = {  # 이름: (통계표, 주기, 항목1, 항목2)
    "cpi": ("901Y009", "M", "0", None),          # 소비자물가지수 총지수
    "ppi": ("404Y014", "M", "*AA", None),        # 생산자물가지수 총지수
    "import_px": ("401Y015", "M", "*AA", "W"),   # 수입물가지수 총지수(원화기준)
    "export": ("901Y118", "M", "T002", None),    # 수출금액
    "gdp": ("200Y110", "Q", "10601", None),      # 실질 GDP(원계열)
    "bsi": ("512Y008", "M", "BA", "C0000"),      # 업황전망 BSI, 제조업 (항목 순서: BSI코드/업종)
}


def fetch(code, cycle, i1, i2):
    start, end = ("200001", "202612") if cycle == "M" else ("2000Q1", "2026Q4")
    items = "/".join(x for x in (i1, i2) if x)
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{KEY}/json/kr/1/10000/{code}/{cycle}/{start}/{end}/{items}"
    j = requests.get(url, timeout=60).json()
    if "StatisticSearch" not in j:
        raise RuntimeError(f"{code}: {j}")
    rows = j["StatisticSearch"]["row"]
    df = pd.DataFrame(rows)
    # 한 항목에 여러 하위계열(예: 통화기준)이 오면 첫 하위계열만 쓴다
    for col in ("ITEM_CODE2", "ITEM_CODE3"):
        if col in df and df[col].nunique() > 1:
            df = df[df[col] == df[col].iloc[0]]
    if cycle == "M":
        idx = pd.to_datetime(df["TIME"], format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        idx = pd.PeriodIndex(df["TIME"].str.replace("Q", "-Q"), freq="Q").to_timestamp(how="end").normalize()
    return pd.Series(pd.to_numeric(df["DATA_VALUE"], errors="coerce").to_numpy(), index=idx).sort_index()


def main():
    out = pd.DataFrame({k: fetch(*v) for k, v in SERIES.items()})
    out.to_parquet(OUT)
    print(out.dropna(how="all").tail(3))
    print({c: str(out[c].dropna().index[0].date()) + "~" + str(out[c].dropna().index[-1].date()) for c in out})


if __name__ == "__main__":
    main()
