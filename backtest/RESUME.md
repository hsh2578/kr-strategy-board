# 재개 메모 (2026-09-23 11:26 일시정지)

사용자가 "진행해줘" 하면 아래 순서대로 바로 이어간다.

## 현재 상태
- 1차 백테스트 60개 완료 → `../전략_백테스트_순위.docx`, `_v2.docx`(슬리피지 민감도 포함), `results/`, `results_slip_low/`, `results_slip_zero/`
- 2차 조사 4건 완료(스윙 25 / 중장기 팩터 26 / 자산배분 26 / 깃허브 코드 40+) — 내용은 대화 기록, 아래 "구현 목록"에 요약
- `strategies.py` 끝에 LT15~LT28(2차 중장기 팩터) **추가·등록 완료**, LT15·LT24만 시험 실행함
- DART 일괄다운로드: `data/dart_bulk/` 43/128개 (2023_1Q~2026_2Q + 2019_4Q_PL). **로그인 세션 필요**
- DART API 연간 전체재무(`dart_collect.py full`): `data/dart/full.parquet` 2015~2017 일부(8,748행). 일괄파일이 다 받아지면 불필요 — 재개 안 해도 됨
- `dart_collect.py main`은 실패(로그 없음) — 일괄파일로 대체하므로 무시

## 재개 순서
1. **DART 일괄파일 나머지 85개**: Playwright로 `https://opendart.fss.or.kr/uat/uia/egovLoginUsr.do` 열기 → 사용자 로그인 →
   list.do POST 로 파일명 목록 → `_(BS|PL|CF)_` 만, `data/dart_bulk/`에 이미 있는 파일명은 제외 →
   `page.waitForEvent('download')` + `location.href='/cmm/downloadFnlttZip.do?fl_nm='+name` + `dl.saveAs()` 로 20개씩.
   (브라우저 스크립트에서 require/fs 불가 → saveAs 사용)
2. **ETF·미국 데이터 확장**(시스템 python, `kr_data.py`의 ETFS 추가 후 `python kr_data.py etfs`):
   195970 195980 182480 130680 280930 304660 329750 226980 147970 167860 138230 136340 102110 101280 195930 192090 225040 152380 273130,
   섹터: 139220~139290 143860 227550 227560 091160 091170 091180 117460 117680 117700 102970 140710 157490 244580.
   미국 신호용(FDR): SPY EFA EEM AGG LQD IEF SHY TIP BND VWO BIL TLT, FRED:UNRATE → `panel/us.parquet`
3. **strategies2.py 작성**(bash heredoc 은 따옴표 문제로 실패 → Write 도구로 별도 파일, strategies.py 맨 끝에서 import)
   - 스윙: SW15/16 기관·외인 주도 거래량5배(20/50일), SW17/18 월말월초 ETF(코스닥150/KODEX200), SW19/20 Double 7s ETF,
     SW21 Connors %b, SW22 누적RSI2 대형주(3슬롯), SW23 Connors R3, SW24 오닐 베이스+거래량2.5배 63일, SW25 거래량 급감,
     KIS 공식 프리셋(trend_filter SMA60+ROC1 / ROC60 모멘텀 / ATR수축→ROC1>3% / SMA20×0.9 이격), 10년 신고가+1ATR 트레일링
   - 중장기: LT29 대형주 12M 모멘텀 40, LT30 +FIP 필터, LT31 Clenow, LT32 연말배당→1월 코스닥 교대,
     easygap RS 로테이션(0.6ret60+0.4ret120, SMA60, KODEX200>200MA), QC 시장상태 모멘텀, consistent momentum(6개월 코호트),
     12-month cycle, 고변동성 내 모멘텀, 모멘텀+회전율
   - 자산배분: 코스피 10개월선+국고채3년, systrader79 평균모멘텀(주식:채권:현금), HAA-Simple, LAA(실업률), DAA1-U1,
     한미 평균모멘텀 스코어, GTAA5, HAA-Balanced 7자산, VAA 공격형, 섹터 ETF 모멘텀 top3, 코스피+나스닥+달러 1:1:1,
     Paired switching, Golden Butterfly, 자산군 모멘텀 top3
4. **DART 패널화**: 일괄 txt(cp949, 탭) → 종목코드·결산기준일·보고서종류별 매출액/매출총이익/영업이익/순이익/자산/부채/자본/영업현금흐름.
   사용가능일 = 결산일 + 90일(사업보고서) / +45일(분기·반기). 연결 우선, 없으면 별도.
   → 신마법공식(GP/A+PBR), F-score/신F, OCF/P, 슈퍼퀄리티, 저자산성장, PIR(영업이익), 밸류업(ROE·FCF)
5. `run_all.py --jobs 3` (새 전략만 캐시 없음 → 자동으로 그것만 실행) → `report.py` → 순위 보고

## 주의
- 데이터 구축은 시스템 python(pandas 2.3, pykrx 동작), 백테스트는 `.venv`(vectorbt, pandas 3.0)
- 엔진 규칙(미래참조 방지, 상한가, 제한폭 2015-06-15)은 `engine.py`·`test_backtest.py`(7개 통과) 참고
