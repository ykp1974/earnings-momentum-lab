@echo off
chcp 65001 >nul
rem ============================================================
rem  バックテスト (ernings_momentum_lab_multi.py)
rem  対象銘柄は同フォルダの ticker_list.txt (実行前に手で入れ替え)
rem  結果は multi_backtest_result.csv に出力されます
rem ============================================================
cd /d "%~dp0"

python ernings_momentum_lab_multi.py %*
if errorlevel 1 (
    echo.
    echo [警告] ernings_momentum_lab_multi.py がエラー終了しました。
    pause
    exit /b 1
)

echo.
echo === 完了: multi_backtest_result.csv を確認してください ===
pause
