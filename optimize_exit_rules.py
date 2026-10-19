# イグジットルール（利確/損切りパラメータ）超高速グリッドサーチ最適化スクリプト
# (multi_backtest_result.csv 読み込み対応版)

import os
import sys
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------------------
# 設定パラメータ
# ------------------------------------------------------------------------------
# 探索するイグジットパラメータのパターン (損切り% / 利確%)
STOP_LOSS_GRID = [-2.0, -3.0, -4.0, -5.0, -7.0, None]  # 損切りライン (Noneは損切りなし)
TAKE_PROFIT_GRID = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, None]  # 利確ライン (Noneは利確なし)

DEFAULT_CSV_PATH = "multi_backtest_result.csv"


# ------------------------------------------------------------------------------
# 1. バックテスト結果CSVの読み込み & 株価履歴の補完
# ------------------------------------------------------------------------------
def load_and_prepare_signals(csv_path: str) -> list:
  """multi_backtest_result.csv を読み込み、保有期間の日足データを取得してシミュレーション用シグナルを構築"""
  if not os.path.exists(csv_path):
    print(f"【エラー】バックテスト結果ファイル '{csv_path}' が見つかりません。")
    return []

  df_csv = pd.read_csv(csv_path)
  if df_csv.empty:
    print("【エラー】CSVファイルが空です。")
    return []

  print(
      f"=== [Step 1] '{csv_path}' から {len(df_csv)} 件のエントリーシグナルを読み込み中 ==="
  )

  signals = []

  # 各銘柄ごとにyfinanceで価格履歴を一括取得して効率化
  unique_tickers = df_csv["ticker"].unique()

  for ticker in unique_tickers:
    # 末尾 .T の自動補正
    formatted_ticker = ticker if ticker.endswith(".T") else f"{ticker}.T"
    stock = yf.Ticker(formatted_ticker)

    df_price = stock.history(period="5y")
    if df_price.empty:
      continue

    df_price = df_price.reset_index()
    df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.tz_localize(None)
    df_price = df_price.sort_values("Date").reset_index(drop=True)

    ticker_rows = df_csv[df_csv["ticker"] == ticker]

    for _, row in ticker_rows.iterrows():
      entry_date_str = str(row["entry_date"])
      exit_date_str = str(row["exit_date"])

      # エントリー日〜イグジット日までの日足データを抽出
      mask = (
          df_price["Date"]
          >= pd.to_datetime(entry_date_str).tz_localize(None)
      ) & (
          df_price["Date"] <= pd.to_datetime(exit_date_str).tz_localize(None)
      )
      hold_df = df_price[mask].copy().reset_index(drop=True)

      if hold_df.empty:
        continue

      signals.append({
          "ticker": formatted_ticker,
          "name": row.get("name", formatted_ticker),
          "earnings_date": row.get("earnings_date", ""),
          "entry_date": entry_date_str,
          "entry_price": row["entry_price"],
          "price_history": hold_df,  # 保有期間の日足データ (Open, High, Low, Close)
      })

  print(
      f" -> 合計 {len(signals)} 件の検証用データセットを準備しました。\n"
  )
  return signals


# ------------------------------------------------------------------------------
# 2. イグジットルールのグリッドサーチシミュレーション
# ------------------------------------------------------------------------------
def simulate_exit_strategy(
    signals: list, stop_loss_pct: float, take_profit_pct: float
) -> dict:
  """特定の（損切り%, 利確%）ルール下での損益パフォーマンスを追跡計算"""
  returns = []

  for sig in signals:
    entry_price = sig["entry_price"]
    price_history = sig["price_history"]

    exit_price = None

    # 保有期間の日数を順番にチェック
    for day_idx, row in price_history.iterrows():
      high_price = row["High"]
      low_price = row["Low"]

      # 1. 損切り判定 (Lowで判定)
      if stop_loss_pct is not None:
        stop_price = entry_price * (1 + stop_loss_pct / 100.0)
        if low_price <= stop_price:
          exit_price = stop_price  # 損切り成行約定
          break

      # 2. 利確判定 (Highで判定)
      if take_profit_pct is not None:
        target_price = entry_price * (1 + take_profit_pct / 100.0)
        if high_price >= target_price:
          exit_price = target_price  # 利確指値約定
          break

    # 利確・損切りに掛からず満了した場合は最終日のCloseで手仕舞い
    if exit_price is None:
      exit_price = price_history.iloc[-1]["Close"]

    ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
    returns.append(ret_pct)

  df_ret = pd.Series(returns)
  win_trades = df_ret[df_ret > 0]

  win_rate = (len(win_trades) / len(df_ret)) * 100.0 if len(df_ret) > 0 else 0
  avg_return = df_ret.mean() if len(df_ret) > 0 else 0
  total_return = df_ret.sum() if len(df_ret) > 0 else 0
  max_drawdown = df_ret.min() if len(df_ret) > 0 else 0

  return {
      "stop_loss": f"{stop_loss_pct}%" if stop_loss_pct else "なし",
      "take_profit": f"+{take_profit_pct}%" if take_profit_pct else "なし",
      "total_trades": len(df_ret),
      "win_rate_pct": round(win_rate, 1),
      "avg_return_pct": round(avg_return, 2),
      "total_return_pct": round(total_return, 2),
      "max_loss_pct": round(max_drawdown, 2),
  }


# ------------------------------------------------------------------------------
# 3. メイン実行エンジン
# ------------------------------------------------------------------------------
def run_optimization():
  # コマンドライン引数からCSVパスを取得（無ければデフォルト名）
  csv_path = (
      sys.argv[1]
      if len(sys.argv) > 1 and sys.argv[1].endswith(".csv")
      else DEFAULT_CSV_PATH
  )

  signals = load_and_prepare_signals(csv_path)
  if not signals:
    print("検証可能なシグナルがありませんでした。")
    return

  print("=== [Step 2] イグジットルール パラメータ最適化 (グリッドサーチ) 開始 ===")

  grid_results = []

  # 全パターンの総当たり計算
  for sl in STOP_LOSS_GRID:
    for tp in TAKE_PROFIT_GRID:
      res = simulate_exit_strategy(signals, sl, tp)
      grid_results.append(res)

  df_grid = pd.DataFrame(grid_results)

  # 平均損益率が高い順にソート
  df_sorted = df_grid.sort_values(
      by="avg_return_pct", ascending=False
  ).reset_index(drop=True)
  df_sorted.index += 1

  print(
      "\n================================================================================"
  )
  print("   【イグジットルール 最適パラメータ パフォーマンスランキング】")
  print(
      "================================================================================"
  )
  print(f"検証イベント数 : 全 {len(signals)} トレード")
  print(df_sorted.to_string())

  best_config = df_sorted.iloc[0]
  print(
      "\n--------------------------------------------------------------------------------"
  )
  print("★【最適解 (Best Strategy)】")
  print(f"  ・損切りライン (Stop Loss)   : {best_config['stop_loss']}")
  print(f"  ・利確ライン   (Take Profit) : {best_config['take_profit']}")
  print(f"  ・勝率                       : {best_config['win_rate_pct']} %")
  print(
      f"  ・平均損益率                 : {best_config['avg_return_pct']:+} %"
  )
  print(f"  ・最大損失率 (1トレード)     : {best_config['max_loss_pct']} %")
  print(
      "--------------------------------------------------------------------------------\n"
  )

  output_csv = "exit_rule_optimization_result.csv"
  df_sorted.to_csv(output_csv, index=False, encoding="utf-8-sig")
  print(f"★ 詳細結果を '{output_csv}' に出力保存しました。")


if __name__ == "__main__":
  run_optimization()