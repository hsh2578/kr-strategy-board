"""외국인·기관 일별 순매수대금(전 종목, 상장폐지 포함) 수집 — pykrx, 시스템 파이썬.

날짜별 전종목 조회라 그날 상장돼 있던 종목이 모두 들어간다(생존편향 없음).
단위: 백만원. 이어받기 가능(받은 날짜는 건너뜀).

1차 실행에서 4스레드 동시 요청이 KRX에 차단돼(빈 응답) 단일 스레드 + 간격 + 실패 시 대기·세션 재생성으로 바꿨다.
하루치 4개 조회(시장 2 × 주체 2) 중 하나라도 실패하면 그날은 저장하지 않고 다음 실행에서 다시 받는다.

    python collect_flow.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, r"C:/Users/hsh/Desktop")
from env_loader import load_env  # noqa: E402

load_env()
from pykrx import stock  # noqa: E402
from pykrx.website.comm import auth  # noqa: E402

DATA = Path(__file__).parent / "data"
OUT = DATA / "flow_raw.parquet"
INVESTORS = {"외국인": "foreign", "기관합계": "inst"}
GAP = 0.5            # 요청 간격(초)
COOLDOWN = 60        # 연속 실패 시 대기(초)


def _renew_session():
    auth._auth_session = None
    auth.get_auth_session()


def one_day(d: pd.Timestamp):
    """성공하면 DataFrame, 하나라도 실패하면 None."""
    ds = d.strftime("%Y%m%d")
    parts = []
    for mkt in ("KOSPI", "KOSDAQ"):
        for inv, name in INVESTORS.items():
            f = None
            for attempt in range(3):
                try:
                    f = stock.get_market_net_purchases_of_equities_by_ticker(ds, ds, mkt, inv)
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(2 * (attempt + 1))
            time.sleep(GAP)
            if f is None or f.empty:
                return None
            parts.append(pd.DataFrame({"date": d, "code": f.index, "field": name,
                                       "net": f["순매수거래대금"].to_numpy() / 1e6}))
    return pd.concat(parts, ignore_index=True)


def main():
    days = pd.read_parquet(DATA / "panel" / "close.parquet").index
    days = days[days >= "2010-06-01"]
    have = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    done = set(have["date"]) if len(have) else set()
    todo = [d for d in days if d not in done]
    print(f"수급: 기존 {len(done)}일 · 대상 {len(todo)}일", flush=True)
    chunks = [have] if len(have) else []
    fails, streak = 0, 0
    for i, d in enumerate(todo, 1):
        df = one_day(d)
        if df is None:
            fails += 1
            streak += 1
            if streak >= 5:
                print(f"  연속 실패 {streak} → {COOLDOWN}초 대기 후 세션 재생성 ({d.date()})", flush=True)
                time.sleep(COOLDOWN)
                _renew_session()
                streak = 0
        else:
            chunks.append(df)
            streak = 0
        if i % 50 == 0 or i == len(todo):
            out = pd.concat(chunks, ignore_index=True)
            out.to_parquet(OUT)
            chunks = [out]
            print(f"  {i}/{len(todo)} 저장 {out['date'].nunique()}일 · 실패 {fails}일", flush=True)


if __name__ == "__main__":
    main()
