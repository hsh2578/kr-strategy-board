"""사이트 데이터 층 최소 검증: 과세 근사, 보유 종목, 페이로드 형식, 기대값 = engine.metrics, 실보유 대조."""
import numpy as np
import pandas as pd
import pytest

import engine as E
import run_all as RA
import site_build as SB
import strategies as S


def test_annual_tax_takes_154_of_positive_year_on_last_day():
    idx = pd.bdate_range("2021-12-28", "2022-12-30")
    r = pd.Series(0.0, index=idx)
    r.loc["2021-12-29"] = 0.10            # 2021년 +10% → 2021 마지막 날 1.54% 차감
    r.loc["2022-03-02"] = -0.05           # 2022년 손실 → 차감 없음
    out = SB.annual_tax(r)
    assert out.loc["2021-12-31"] == pytest.approx(-0.10 * 0.154)
    assert out.loc["2022-12-30"] == 0.0
    assert out.loc["2021-12-29"] == 0.10


def test_cycle_from_strategy_name():
    assert SB.cycle("코스피·나스닥100·달러 1:1:1, 분기") == "분기"
    assert SB.cycle("신F-스코어+저PBR 25종목, 연1회") == "연 1회"
    assert SB.cycle("LAA 한국판") == "월간"


def test_holdings_aa17_three_etfs_sum_one():
    h, last = SB.holdings("AA17")
    assert len(h) == 3 and h["weight"].sum() == pytest.approx(1.0)
    assert set(h["code"]) == {"069500", "133690", "138230"}
    assert isinstance(last, pd.Timestamp)


@pytest.mark.parametrize("sid", ["AA17", "LT32", "AA10"])
def test_holdings_match_engine_actual_positions(sid):
    """같은 코드 비교가 아니라 엔진이 실제 체결한 마지막 보유와 대조."""
    res = S.REG[sid]["fn"]()
    px = {k.lower(): v for k, v in res["px"].items()} if res.get("px") is not None else RA.stock_px()
    idx = px["close"].index
    pf = E.rebalance(res["weights"], px, pd.Series(0.0, index=idx),
                     pd.DataFrame(0.0, index=idx, columns=px["close"].columns))
    held = pf.asset_value(group_by=False).iloc[-1]
    actual = set(held[held > 1e-9].index)
    h, _ = SB.holdings(sid)
    assert set(h["code"]) == actual


@pytest.fixture(scope="module")
def payload():
    return SB.payload()


def test_payload_series_aligned(payload):
    n = len(payload["dates"])
    ids = SB.load()["site"].index.tolist() + list(SB.BENCH)
    assert set(payload["series"]) == set(ids)
    assert all(len(v) == n for v in payload["series"].values()) and len(payload["rf"]) == n
    assert all(x is None or np.isfinite(x) for v in payload["series"].values() for x in v)


def test_expected_equals_engine_metrics(payload):
    d = pd.Series(payload["series"]["AA17"], index=pd.to_datetime(payload["dates"]), dtype=float).dropna()
    m = E.metrics(d.loc["2011":"2024"])
    e = payload["expected"]["AA17"]["2011-2024"]
    assert e["CAGR"] == pytest.approx(m["CAGR"], abs=1e-9) and e["Sharpe"] == pytest.approx(m["Sharpe"], abs=1e-9)


def test_tiers_use_exact_2011_2024_window():
    """등급 판정 수치는 정확히 2011-01-01~2024-12-31 (전략 시작이 2010이어도 2010 제외)."""
    t = SB.build_tiers(write=False)
    d = pd.read_parquet(SB.RES / "daily.parquet")
    for sid in t.index[t.tier == "1 검증 통과"]:
        m = E.metrics(d[sid].dropna().loc["2011":"2024"])
        assert t.at[sid, "S24"] == pytest.approx(m["Sharpe"]) and m["Sharpe"] >= 0.5 and t.at[sid, "T24"] > 2
    k = E.metrics(d["H01"].dropna().loc["2011":"2024"])
    for sid in t.index[t.tier == "4 수익 플러스·지수 이하"]:
        assert t.at[sid, "C24"] <= k["CAGR"] and t.at[sid, "S24"] <= k["Sharpe"] and t.at[sid, "S24"] > 0


def test_every_site_strategy_has_description(payload):
    missing = [i for i, m in payload["meta"].items() if i not in SB.BENCH and not (m.get("desc") and m.get("short"))]
    assert not missing, missing
