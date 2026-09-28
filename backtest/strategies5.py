"""DART 공시 이벤트 전략(OpenDART list.json, 2015~). 접수 시각이 없어 접수일 장 마감 후 알았다고 보고
다음 거래일 시가에 진입한다(휴일 접수는 직전 거래일 신호 → 다음 거래일 시가).

잠정실적 서프라이즈는 목록에 실적 숫자가 없어 보류(ponytail: 공시 본문 2.5만 건 파싱 필요, 할 때 추가).
"""
import numpy as np
import pandas as pd

import kr_data as K
from strategies import d, stock, strat

BAD = ("rights_offering", "cb", "bw", "admin_issue", "unfaithful", "audit_opinion")
SINCE = "2015-02-01"


def _events(kinds, exclude_word=None):
    ev = pd.read_parquet(K.PANEL.parent / "dart" / "disclosures.parquet")
    ev = ev[ev.event.isin(kinds)]
    if exclude_word:
        ev = ev[~ev.report_nm.str.contains(exclude_word, regex=False)]
    idx = d.C.index
    dt = pd.to_datetime(ev.rcept_dt)
    pos = idx.searchsorted(dt, side="right") - 1          # 접수일 이하 마지막 거래일
    ev = ev.assign(day=idx[np.clip(pos, 0, None)])[pos >= 0]
    sig = pd.crosstab(ev.day, ev.stock_code).gt(0)
    return sig.reindex(index=idx, columns=d.C.columns, fill_value=False)


def _bad_recent(days=250):
    """최근 약 1년 안에 유상증자·CB·BW·관리종목·불성실공시·감사의견 공시가 있었던 종목."""
    return _events(BAD).astype(float).rolling(days, min_periods=1).max().gt(0)


def _dip(n=10):
    return d.C / d.C.shift(n) < 1


@strat("EV01", "공시", "자사주 직접취득 결정 → 다음날 시가, 20거래일 보유", "자사주 공시효과(한국 다수 연구)", since=SINCE,
       assume="정정공시 제외, 슬롯 10")
def ev01(): return stock(_events(["buyback_direct"], "정정"), hold=20)


@strat("EV02", "공시", "자사주 직접취득 + 공시 전 10일 하락 → 20거래일", "가격방어 목적 매입 효과", since=SINCE)
def ev02(): return stock(_events(["buyback_direct"], "정정") & _dip(), hold=20)


@strat("EV03", "공시", "자사주 신탁계약 체결 → 다음날 시가, 20거래일 보유", "자사주 신탁 공시효과", since=SINCE)
def ev03(): return stock(_events(["buyback_trust"], "정정"), hold=20)


@strat("EV04", "공시", "EV02 + 최근 1년 악재공시 종목 제외", "공통 제외 필터", since=SINCE)
def ev04(): return stock(_events(["buyback_direct"], "정정") & _dip() & ~_bad_recent(), hold=20)


@strat("EV05", "공시", "EV03 + 최근 1년 악재공시 종목 제외", "공통 제외 필터", since=SINCE)
def ev05(): return stock(_events(["buyback_trust"], "정정") & ~_bad_recent(), hold=20)


# ─── 잠정실적 서프라이즈(컨센서스 이력이 없어 전년동기 대비로 정의, 결과 보기 전 고정) ───
def _prelim_surprise():
    """영업이익 전년동기 대비 +30% 이상(또는 흑자전환) 이면서 매출도 전년동기보다 증가한 잠정실적 공시."""
    pr = pd.read_parquet(K.PANEL.parent / "dart" / "prelim.parquet")
    ev = pd.read_parquet(K.PANEL.parent / "dart" / "disclosures.parquet")[["rcept_no", "stock_code", "rcept_dt"]]
    pr = pr.merge(ev, on="rcept_no").dropna(subset=["op_cur", "op_yago", "rev_cur", "rev_yago"])
    op_up = ((pr.op_yago > 0) & (pr.op_cur >= 1.3 * pr.op_yago)) | ((pr.op_yago <= 0) & (pr.op_cur > 0))
    pr = pr[op_up & (pr.rev_cur > pr.rev_yago)]
    idx = d.C.index
    pos = idx.searchsorted(pd.to_datetime(pr.rcept_dt), side="right") - 1
    pr = pr.assign(day=idx[np.clip(pos, 0, None)])[pos >= 0]
    return pd.crosstab(pr.day, pr.stock_code).gt(0).reindex(index=idx, columns=d.C.columns, fill_value=False)


if (K.PANEL.parent / "dart" / "prelim.parquet").exists():
    @strat("EV06", "공시", "잠정실적 서프라이즈(영업이익 YoY +30%↑ 또는 흑전, 매출 YoY↑) → 다음날 시가, 20거래일",
           "PEAD·잠정실적 공시효과(한국 연구 다수)", since=SINCE, assume="컨센서스 대신 전년동기 대비")
    def ev06(): return stock(_prelim_surprise(), hold=20)

    @strat("EV07", "공시", "EV06 과 같은 신호, 10거래일 보유", "PEAD 단기", since=SINCE)
    def ev07(): return stock(_prelim_surprise(), hold=10)

    @strat("EV08", "공시", "EV06 + 최근 1년 악재공시 종목 제외", "공통 제외 필터", since=SINCE)
    def ev08(): return stock(_prelim_surprise() & ~_bad_recent(), hold=20)
