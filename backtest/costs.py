"""한국 주식 거래비용. vectorbt 의 fees / slippage 인자로 그대로 넣을 배열을 만든다.

vectorbt 는 fees 를 매수·매도에 같은 비율로 부과하므로, 매도에만 붙는 거래세는
양쪽에 반씩 나눠 넣는다(왕복 합계는 동일).
"""
import os

import numpy as np
import pandas as pd

COMMISSION = 0.00015  # 편도 수수료 0.015%

# 매도 거래세(농특세 포함, 주식). ETF 는 0.
# 2026: 2025 세법개정으로 0.20% 환원 — 구현 시점 확인값
SELL_TAX = {
    2010: 0.0030, 2011: 0.0030, 2012: 0.0030, 2013: 0.0030, 2014: 0.0030,
    2015: 0.0030, 2016: 0.0030, 2017: 0.0030, 2018: 0.0030, 2019: 0.0030,
    2020: 0.0025, 2021: 0.0023, 2022: 0.0023, 2023: 0.0020, 2024: 0.0018,
    2025: 0.0015, 2026: 0.0020,
}

# 편도 슬리피지: 전일 시총 기준 (HollyKR 과 동일 구간)
SLIP_TIERS = [(5e12, 0.001), (1e12, 0.002), (0, 0.005)]
if os.environ.get("BT_SLIP"):  # 민감도 테스트: BT_SLIP="0.0005,0.001,0.002"
    SLIP_TIERS = [(cap, float(v)) for (cap, _), v in zip(SLIP_TIERS, os.environ["BT_SLIP"].split(","))]
ETF_SLIP = 0.0005


def fee_rate(index: pd.DatetimeIndex, etf: bool = False) -> pd.Series:
    """편도 수수료 + 거래세/2 (날짜별)."""
    tax = np.zeros(len(index)) if etf else index.year.map(SELL_TAX).to_numpy(float)
    return pd.Series(COMMISSION + tax / 2, index=index)


def slippage(prev_mktcap: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(SLIP_TIERS[-1][1], index=prev_mktcap.index, columns=prev_mktcap.columns)
    for cap, s in reversed(SLIP_TIERS[:-1]):  # 작은 구간부터 덮어써서 큰 시총이 최종
        out = out.mask(prev_mktcap >= cap, s)
    return out


def round_trip(year: int, etf: bool = False) -> float:
    """왕복 비용(슬리피지 제외). 테스트·보고서용."""
    return 2 * COMMISSION + (0.0 if etf else SELL_TAX[year])
