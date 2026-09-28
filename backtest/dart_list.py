"""OpenDART 공시 목록(list.json) 수집 — 이벤트 전략·제외 필터용. 이어받기 가능.

공시유형 B(주요사항보고)·I(거래소공시)를 분기 창으로 전부 받아, 필요한 보고서명만 남긴다.
접수일(rcept_dt)만 있고 시각은 없으므로 진입은 접수일 다음 거래일 시가로 한다(보수적).

    python dart_list.py
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, r"C:/Users/hsh/Desktop")
from env_loader import load_env  # noqa: E402

load_env()
KEY = os.environ["DART_API_KEY"]
OUT = Path(__file__).parent / "data" / "dart" / "disclosures.parquet"
KEEP = [  # 보고서명 포함 키워드 → 이벤트 이름
    ("자기주식취득신탁계약체결결정", "buyback_trust"),
    ("자기주식취득결정", "buyback_direct"),
    ("영업(잠정)실적", "prelim_earnings"),
    ("유상증자결정", "rights_offering"),
    ("전환사채권발행결정", "cb"),
    ("신주인수권부사채권발행결정", "bw"),
    ("관리종목지정", "admin_issue"),
    ("불성실공시법인지정", "unfaithful"),
    ("감사의견", "audit_opinion"),
    ("주권매매거래정지", "trading_halt"),
]


def page(ty, bgn, end, no):
    for i in range(4):
        try:
            j = requests.get("https://opendart.fss.or.kr/api/list.json", timeout=60, params={
                "crtfc_key": KEY, "pblntf_ty": ty, "bgn_de": bgn, "end_de": end,
                "page_no": no, "page_count": 100}).json()
            if j.get("status") == "020":
                raise SystemExit("DART 일일 한도 초과 — 내일 이어받기")
            return j
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001
            time.sleep(2 + 3 * i)
    return {}


def window(args):
    ty, bgn, end = args
    first = page(ty, bgn, end, 1)
    if first.get("status") != "000":
        return pd.DataFrame()
    rows = list(first["list"])
    for no in range(2, int(first["total_page"]) + 1):
        rows += page(ty, bgn, end, no).get("list", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["event"] = None
    for kw, name in KEEP:
        hit = df["report_nm"].str.replace(" ", "").str.contains(kw, regex=False) & df["event"].isna()
        df.loc[hit, "event"] = name
    return df[df["event"].notna() & (df["stock_code"].str.len() == 6)][
        ["corp_code", "stock_code", "corp_name", "corp_cls", "report_nm", "rcept_no", "rcept_dt", "event"]]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    have = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    months = pd.period_range("2015-01", pd.Timestamp.today().strftime("%Y-%m"), freq="M")
    done = set(have["window"]) if len(have) else set()
    jobs = [(ty, m.start_time.strftime("%Y%m%d"), m.end_time.strftime("%Y%m%d"))
            for m in months for ty in ("B", "I") if f"{ty}{m}" not in done]
    print(f"공시 목록: 대상 {len(jobs)}개 창(월×유형)", flush=True)
    chunks = [have] if len(have) else []
    with ThreadPoolExecutor(3) as ex:
        for i, (job, df) in enumerate(zip(jobs, ex.map(window, jobs)), 1):
            df = df.assign(window=f"{job[0]}{pd.Period(job[1][:6], 'M')}")
            chunks.append(df)
            if i % 20 == 0 or i == len(jobs):
                out = pd.concat(chunks, ignore_index=True)
                out.to_parquet(OUT)
                chunks = [out]
                print(f"  {i}/{len(jobs)} · 이벤트 {len(out):,}건", flush=True)
    out = pd.read_parquet(OUT)
    print(out["event"].value_counts().to_string())


if __name__ == "__main__":
    main()
