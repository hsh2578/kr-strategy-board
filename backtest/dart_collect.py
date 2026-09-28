"""OpenDART 재무 수집 (2015 사업연도~). 시스템 파이썬으로 실행.

1) main : 주요계정(fnlttMultiAcnt, 100사 묶음) — 분기·연간, 전 종목
2) full : 전체재무제표(fnlttSinglAcntAll) 연간 — 매출총이익·영업활동현금흐름만 추출,
          해당 연도 말 시총 1,000억 이상 종목만(호출 한도 20,000/일)
모든 행에 접수일(rcept_dt)을 남겨 백테스트에서 공시 전 사용을 막는다.

    python dart_collect.py main
    python dart_collect.py full
"""
import io
import os
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, r"C:/Users/hsh/Desktop")
from env_loader import load_env  # noqa: E402

load_env()
KEY = os.environ["DART_API_KEY"]
API = "https://opendart.fss.or.kr/api"
DATA = Path(__file__).parent / "data"
OUT = DATA / "dart"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS = {"11013": "Q1", "11012": "Q2", "11014": "Q3", "11011": "Q4"}
YEARS = range(2015, 2027)


def get(path, **params):
    for i in range(4):
        try:
            r = requests.get(f"{API}/{path}", params={"crtfc_key": KEY, **params}, timeout=60)
            if path.endswith(".json"):
                j = r.json()
                if j.get("status") == "020":        # 요청 제한 초과
                    raise RuntimeError("DART 일일 한도 초과")
                return j
            return r.content
        except RuntimeError:
            raise
        except Exception:
            time.sleep(2 + i * 3)
    return {}


def corp_codes() -> pd.DataFrame:
    f = OUT / "corp_codes.parquet"
    if f.exists():
        return pd.read_parquet(f)
    z = zipfile.ZipFile(io.BytesIO(get("corpCode.xml")))
    root = ET.fromstring(z.read(z.namelist()[0]))
    rows = [{c.tag: (c.text or "").strip() for c in el} for el in root.iter("list")]
    df = pd.DataFrame(rows)
    df = df[df["stock_code"].str.len() == 6][["corp_code", "stock_code", "corp_name"]]
    df.to_parquet(f)
    return df


def our_codes() -> pd.DataFrame:
    meta = pd.read_parquet(DATA / "panel" / "meta.parquet")
    cc = corp_codes()
    return cc[cc["stock_code"].isin(meta.index)]


def main_accounts():
    f = OUT / "main.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    done = set(zip(have["bsns_year"], have["reprt_code"])) if len(have) else set()
    codes = our_codes()["corp_code"].tolist()
    batches = [codes[i:i + 100] for i in range(0, len(codes), 100)]
    rows = [have] if len(have) else []
    for y in YEARS:
        for rc in REPORTS:
            if (str(y), rc) in done:
                continue
            with ThreadPoolExecutor(4) as ex:
                res = list(ex.map(lambda b: get("fnlttMultiAcnt.json", corp_code=",".join(b),
                                               bsns_year=str(y), reprt_code=rc), batches))
            got = [pd.DataFrame(j["list"]) for j in res if j.get("status") == "000"]
            if got:
                rows.append(pd.concat(got, ignore_index=True))
                pd.concat(rows, ignore_index=True).to_parquet(f)
            print(f"main {y} {REPORTS[rc]}: {sum(len(g) for g in got)}행", flush=True)


FULL_KEEP = {"매출총이익", "영업활동현금흐름", "영업활동으로인한현금흐름", "영업활동으로 인한 현금흐름",
             "매출원가", "수익(매출액)", "매출액"}


def _full_one(args):
    corp, y = args
    for fs in ("CFS", "OFS"):
        j = get("fnlttSinglAcntAll.json", corp_code=corp, bsns_year=str(y), reprt_code="11011", fs_div=fs)
        if j.get("status") == "000":
            df = pd.DataFrame(j["list"])
            ids = df["account_id"].fillna("")
            nm = df["account_nm"].str.replace(" ", "")
            keep = (ids.isin(["ifrs-full_GrossProfit", "ifrs-full_CashFlowsFromUsedInOperatingActivities",
                              "ifrs_GrossProfit", "ifrs_CashFlowsFromUsedInOperatingActivities"])
                    | nm.isin({k.replace(" ", "") for k in FULL_KEEP}))
            out = df[keep][["corp_code", "bsns_year", "sj_div", "account_id", "account_nm",
                            "thstrm_amount", "rcept_no"]].copy()
            out["fs_div"] = fs
            return out
    return None


def full_statements():
    f = OUT / "full.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    done = set(zip(have["corp_code"], have["bsns_year"].astype(int))) if len(have) else set()
    cap = pd.read_parquet(DATA / "panel" / "mktcap.parquet")
    cc = our_codes().set_index("stock_code")["corp_code"]
    todo = []
    for y in YEARS:
        ye = cap.loc[:f"{y}-12-31"].iloc[-1] if y < 2026 else cap.iloc[-1]
        stocks = ye[ye >= 1e11].index.intersection(cc.index)
        todo += [(cc[s], y) for s in stocks if (cc[s], y) not in done]
    print(f"full: 대상 {len(todo)}건", flush=True)
    rows = [have] if len(have) else []
    try:
        with ThreadPoolExecutor(6) as ex:
            for i, out in enumerate(ex.map(_full_one, todo), 1):
                if out is not None:
                    rows.append(out)
                if i % 500 == 0 or i == len(todo):
                    pd.concat(rows, ignore_index=True).to_parquet(f)
                    print(f"  {i}/{len(todo)}", flush=True)
    finally:
        if rows:
            pd.concat(rows, ignore_index=True).to_parquet(f)


if __name__ == "__main__":
    {"main": main_accounts, "full": full_statements}[sys.argv[1]]()
