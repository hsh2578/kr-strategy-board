"""잠정실적 공시 본문(OpenDART document.xml) → 매출·영업이익 당해실적과 전년동기 값. 이어받기 가능.

대상: 유니버스(접수일 시총 ≥ 1,000억) 잠정실적 공시, 월별·기재정정·첨부·자회사 공시 제외.
원문 텍스트(표 부분)도 같이 저장해 파서를 고쳐도 다시 받지 않는다.

    python dart_prelim.py          # 시스템 python, 일일 한도(020) 걸리면 멈춤 → 다음 날 다시 실행
"""
import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, r"C:/Users/hsh/Desktop")
from env_loader import load_env  # noqa: E402

load_env()
KEY = os.environ["DART_API_KEY"]
DIR = Path(__file__).parent / "data" / "dart"
OUT = DIR / "prelim.parquet"


def targets():
    import kr_data as K
    d = pd.read_parquet(DIR / "disclosures.parquet")
    p = d[d.event == "prelim_earnings"]
    nm = p.report_nm.str.replace(" ", "")
    p = p[~nm.str.contains("월별|기재정정|첨부|자회사")]
    cap = K.load("mktcap")
    pos = cap.index.searchsorted(pd.to_datetime(p.rcept_dt), side="right") - 1
    col = {c: i for i, c in enumerate(cap.columns)}
    v = cap.to_numpy()
    p = p.assign(cap=[v[i, col[c]] if c in col and i >= 0 else float("nan") for i, c in zip(pos, p.stock_code)])
    return p[p.cap >= 1e11]


def _num(s):
    s = s.split("(")[0].strip() if re.match(r"^-?[\d,.]+\s*\(", s) else s.strip()
    if not re.fullmatch(r"\(?-?[\d,]+(?:\.\d+)?\)?", s):
        return None
    neg = s.startswith("(") or s.startswith("-")
    x = float(s.strip("()-").replace(",", ""))
    return -x if neg else x


def rows_of(html):
    """표의 '항목 | 당해실적 | ...' 행만 텍스트로."""
    import lxml.html as H
    out = []
    for tr in H.fromstring(html).iter("tr"):
        cells = [" ".join(c.text_content().split()) for c in tr if c.tag in ("td", "th")]
        if len(cells) >= 3 and cells[1] == "당해실적":
            out.append(" | ".join(cells))
    return "\n".join(out)


def parse(text):
    """셀 배치 3종: 당기·전기·전기대비·전년동기·전년동기대비(5칸) 또는 전환여부 칸 포함(7칸)."""
    out = {}
    for line in text.split("\n"):
        cells = line.split(" | ")
        key = {"매출액": "rev", "영업이익": "op", "당기순이익": "ni"}.get(cells[0])
        vals = cells[2:]
        if not key or f"{key}_cur" in out or len(vals) not in (5, 7):
            continue
        out[f"{key}_cur"] = _num(vals[0])
        out[f"{key}_yago"] = _num(vals[4] if len(vals) == 7 else vals[3])
    return out


def fetch(rcept_no):
    for i in range(4):
        try:
            r = requests.get("https://opendart.fss.or.kr/api/document.xml", timeout=60,
                             params={"crtfc_key": KEY, "rcept_no": rcept_no})
            if r.content[:2] != b"PK":
                if b"020" in r.content[:300]:
                    raise SystemExit("DART 일일 한도 초과 — 내일 이어받기")
                return {"rcept_no": rcept_no, "err": r.content[:120].decode("utf-8", "ignore")}
            z = zipfile.ZipFile(io.BytesIO(r.content))
            b = z.read(z.namelist()[0])
            try:
                html = b.decode("utf-8")
            except UnicodeDecodeError:
                html = b.decode("cp949", "ignore")
            text = rows_of(html)
            return {"rcept_no": rcept_no, "text": text, **parse(text)}
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001
            time.sleep(2 + 3 * i)
    return {"rcept_no": rcept_no, "err": "fetch failed"}


def main(limit=None):
    have = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["rcept_no"])
    done = set(have.loc[have.get("err", pd.Series(index=have.index, dtype=object)).isna(), "rcept_no"]) if len(have) else set()
    todo = [r for r in targets().rcept_no if r not in done][:limit]
    print(f"잠정실적 본문: 남은 {len(todo)}건", flush=True)
    rows, n = [], 0
    try:
        with ThreadPoolExecutor(3) as ex:
            for row in ex.map(fetch, todo):
                rows.append(row)
                n += 1
                if n % 200 == 0:
                    have = _save(have, rows); rows = []
                    print(f"  {n}/{len(todo)}", flush=True)
    finally:
        have = _save(have, rows)
        print(f"저장 {len(have)}건 · 파싱 성공(영업이익) {have.get('op_cur', pd.Series()).notna().sum()}건", flush=True)


def _save(have, rows):
    if rows:
        have = pd.concat([have[~have.rcept_no.isin([r["rcept_no"] for r in rows])], pd.DataFrame(rows)],
                         ignore_index=True)
        have.to_parquet(OUT)
    return have


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
