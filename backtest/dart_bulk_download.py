"""DART 재무정보 일괄다운로드(로그인 필요). 브라우저 창이 뜨면 사용자가 로그인 → 자동 감지 후 BS/PL/CF 전부 받는다.
이미 받은 파일은 건너뛴다. 로그인 정보는 저장하지 않는다.

    .venv/Scripts/python dart_bulk_download.py
"""
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "data" / "dart_bulk"
BASE = "https://opendart.fss.or.kr"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in OUT.glob("*.zip")}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, channel="chrome") if _has_chrome() else pw.chromium.launch(headless=False)
        page = browser.new_context(accept_downloads=True).new_page()
        page.goto(f"{BASE}/uat/uia/egovLoginUsr.do")
        print("브라우저 창에서 DART 로그인해 주세요 (최대 20분 대기)", flush=True)
        for _ in range(1200):
            html = page.request.post(f"{BASE}/disclosureinfo/fnltt/dwld/list.do", form={}).text()
            files = sorted(set(re.findall(r"download_ext002\('\d{4}','\w+', '\w+', '([^']+)'\)", html)))
            if files and _logged_in(page):
                break
            time.sleep(1)
        else:
            raise SystemExit("로그인 대기 시간 초과")
        todo = [f for f in files if re.search(r"_(BS|PL|CF)_", f) and f not in have]
        print(f"로그인 확인 · 받을 파일 {len(todo)}개", flush=True)
        page.goto(f"{BASE}/disclosureinfo/fnltt/dwld/main.do")
        fail = []
        for i, name in enumerate(todo, 1):
            try:
                with page.expect_download(timeout=180_000) as dl:
                    page.evaluate("n => { window.location.href = '/cmm/downloadFnlttZip.do?fl_nm=' + n }", name)
                dl.value.save_as(OUT / name)
            except Exception as e:  # noqa: BLE001
                fail.append(name)
                print(f"  실패 {name}: {type(e).__name__}", flush=True)
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
        print(f"완료 · 실패 {len(fail)}개 {fail}", flush=True)
        browser.close()


def _logged_in(page) -> bool:
    """로그인해야 열리는 회원정보 페이지가 로그인 화면으로 튕기지 않으면 로그인 상태."""
    try:
        r = page.request.get(f"{BASE}/mng/selectUserInfoView.do")
        return r.ok and "egovLoginUsr" not in r.url and "egovLoginUsr" not in r.text()[:5000]
    except Exception:  # noqa: BLE001
        return False


def _has_chrome() -> bool:
    return Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe").exists()


if __name__ == "__main__":
    main()
