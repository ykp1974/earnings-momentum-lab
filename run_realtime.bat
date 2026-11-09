@echo off
chcp 65001 >nul
rem ============================================================
rem  当日決算の判定 & S株買い候補抽出 (ernings_momentum_realtime.py)
rem  基準日は引数で指定 (省略時は本日)。.env の書き換えは不要です。
rem    run_realtime.bat                       ... 本日分
rem    run_realtime.bat --date 2026-11-06     ... 日付を指定
rem    run_realtime.bat --tickers 3697,7730   ... テスト用に銘柄を指定
rem ============================================================
cd /d "%~dp0"

python ernings_momentum_realtime.py %*
if errorlevel 1 (
    echo.
    echo [警告] ernings_momentum_realtime.py がエラー終了しました。
    pause
    exit /b 1
)

echo.
echo === 完了: realtime_s_stock_candidates_日付.csv を確認してください ===
pause
