"""KRX 일봉 원천 수집. find-a/krx_data.py 의 fetch_day 를 그대로 재사용한다.

KOSPI 는 find-a 캐시를 복사해 이어받고, KOSDAQ 은 새로 받는다(상장폐지 포함, 날짜별 전 종목).
하루 1회 호출이 약 2초라 8스레드로 병렬 수집한다. 이미 받은 날짜는 건너뛴다.
    python collect.py
"""
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, r"C:/Users/hsh/Desktop")
sys.path.insert(0, r"C:/Users/hsh/Desktop/find-a")
from env_loader import load_env  # noqa: E402
import krx_data  # noqa: E402

DATA = Path(__file__).parent / "data"
FIND_A_KOSPI = Path(r"C:/Users/hsh/Desktop/find-a/data/krx_raw_kospi.parquet")


ETF_URL = "http://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd"


def fetch_etf_day(date: str, key: str):
    """ETF 하루치 전 종목. 형식은 주식 원천과 같게 맞춘다."""
    import requests
    for i in range(3):
        try:
            r = requests.get(ETF_URL, headers={"AUTH_KEY": key}, params={"basDd": date}, timeout=30)
            rows = r.json().get("OutBlock_1", [])
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            for c in krx_data.NUM_COLS:
                if c in df:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
            df["BAS_DD"] = pd.to_datetime(df["BAS_DD"], format="%Y%m%d")
            return df[[c for c in ["BAS_DD", "ISU_CD", "ISU_NM"] + krx_data.NUM_COLS if c in df]]
        except Exception:
            import time
            time.sleep(1 + i)
    return pd.DataFrame()


def collect(market: str, key: str, workers: int = 8):
    path = DATA / f"krx_raw_{market.lower()}.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    done = set(pd.to_datetime(have["BAS_DD"]).dt.normalize()) if len(have) else set()
    days = [d for d in pd.bdate_range(krx_data.EARLIEST, pd.Timestamp.today()) if d not in done]
    print(f"{market}: 기존 {len(done)}일 · 수집 대상 {len(days)}일", flush=True)
    chunks = [have] if len(have) else []
    with ThreadPoolExecutor(workers) as ex:
        fetch = (lambda d: fetch_etf_day(d.strftime("%Y%m%d"), key)) if market == "ETF" else             (lambda d: krx_data.fetch_day(d.strftime("%Y%m%d"), key, market))
        for i, df in enumerate(ex.map(fetch, days), 1):
            if len(df):
                chunks.append(df)
            if i % 250 == 0 or i == len(days):
                out = pd.concat(chunks, ignore_index=True).drop_duplicates(["BAS_DD", "ISU_CD"])
                out.sort_values(["BAS_DD", "ISU_CD"]).to_parquet(path, index=False)
                chunks = [out]
                print(f"  {i}/{len(days)} 저장 {out['BAS_DD'].nunique()}일", flush=True)


def main():
    load_env()
    key = os.environ["KRX_API_KEY"]
    DATA.mkdir(exist_ok=True)
    if not (DATA / "krx_raw_kospi.parquet").exists():
        shutil.copy(FIND_A_KOSPI, DATA / "krx_raw_kospi.parquet")
    for market in (sys.argv[1:] or ["KOSPI", "KOSDAQ"]):
        collect(market, key)


if __name__ == "__main__":
    main()
