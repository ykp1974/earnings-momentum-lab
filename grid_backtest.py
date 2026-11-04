# 決算モメンタム パラメータグリッド探索スクリプト
# LLM評価は (モデル, 銘柄, 決算日) ごとに1回だけ実行してキャッシュし、
# 閾値・保有日数・検証年数の組合せはキャッシュ済みスコアからオフラインで再計算する。

import itertools
import json
import os

import pandas as pd
import yfinance as yf

# LLM 評価関数・設定は既存スクリプトを再利用する
from ernings_momentum_lab_multi import (
    ACTIVE_MODEL,
    LLM_PROVIDER,
    TICKER_LIST_FILE,
    evaluate_earnings_with_llm,
)

import time

# ------------------------------------------------------------------------------
# グリッド設定（必要に応じて編集）
# ------------------------------------------------------------------------------
THRESHOLDS = [0, 70, 75, 80, 85, 90]  # 0 は「全決算を買う」比較対象
ENTRY_LAG = 1  # 反応日(決算翌営業日)から何営業日後の始値で買うか（1=反応確認型）
HOLD_DAYS_LIST = [3, 4, 5, 7, 10]
YEARS_LIST = [3, 5]
MAX_YEARS = max(YEARS_LIST)
MIN_TRADES = 20  # ランキング対象とする最小トレード数

CACHE_FILE = "llm_score_cache.json"
EVENTS_FILE = "grid_events.csv"
GRID_RESULT_FILE = "grid_result.csv"


# ------------------------------------------------------------------------------
# キャッシュ入出力
# ------------------------------------------------------------------------------
def load_cache() -> dict:
  if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
      return json.load(f)
  return {}


def save_cache(cache: dict) -> None:
  with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=1)


def load_tickers() -> list[str]:
  tickers = []
  with open(TICKER_LIST_FILE, "r", encoding="utf-8") as f:
    for line in f:
      code = line.strip()
      if code and not code.startswith("#"):
        tickers.append(code if code.endswith(".T") else f"{code}.T")
  return tickers


# ------------------------------------------------------------------------------
# 1. 全決算イベントの収集とスコア取得（LLMはキャッシュ未登録分のみ）
# ------------------------------------------------------------------------------
def collect_events(ticker: str, cache: dict, bench) -> list[dict]:
  """銘柄の全決算イベントについて、スコアと各保有日数のリターンを算出して返す"""
  stock = yf.Ticker(ticker)
  df = stock.history(period=f"{MAX_YEARS}y")
  if df.empty:
    print("  └─ ⚠️ 株価データなし。スキップ")
    return []
  df = df.reset_index()
  df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
  df = df.sort_values("Date").reset_index(drop=True)

  # Yahoo 側の一時的な失敗があるため、例外時は待って再試行する
  ed = None
  for attempt in range(3):
    try:
      ed = stock.earnings_dates
      break
    except Exception as e:
      print(f"  └─ ⚠️ 決算日データ取得失敗 ({type(e).__name__}: {e})。再試行 {attempt + 1}/3")
      time.sleep(10 * (attempt + 1))
  else:
    print("  └─ ⚠️ 決算日データ取得失敗。スキップ")
    return []
  if ed is None or ed.empty:
    print("  └─ ⚠️ 決算日データなし。スキップ")
    return []
  ed = ed.reset_index()
  ed["Earnings Date"] = pd.to_datetime(ed["Earnings Date"]).dt.tz_localize(None)
  ed = ed[ed["Earnings Date"] <= df["Date"].max()]

  events = []
  for _, row in ed.iterrows():
    e_date = row["Earnings Date"].normalize()
    e_str = e_date.strftime("%Y-%m-%d")
    before = df[df["Date"] <= e_date]
    after = df[df["Date"] > e_date]
    if before.empty or after.empty:
      continue

    b_idx, a_idx = before.index[-1], after.index[0]
    p_before, p_after = df.loc[b_idx, "Close"], df.loc[a_idx, "Close"]
    v_before, v_after = df.loc[b_idx, "Volume"], df.loc[a_idx, "Volume"]
    chg = (p_after - p_before) / p_before * 100.0 if p_before > 0 else 0
    vol_ratio = v_after / v_before if v_before > 0 else 1.0

    key = f"{ACTIVE_MODEL}|{ticker}|{e_str}"
    if key in cache:
      score = cache[key]["score"]
    else:
      text = (
          f"【決算発表日】: {e_str}\n"
          f"【決算直後の市場反応】: 株価変化率 {chg:+.2f}%, 出来高変動"
          f" {vol_ratio:.1f}倍\n"
          f"【株価動向サマリー】: 発表直前の株価 {p_before:.1f}円 ➔ 発表直後の株価"
          f" {p_after:.1f}円"
      )
      score, reason = evaluate_earnings_with_llm(ticker, e_str, text)
      if reason.startswith("エラー発生"):
        continue  # エラーはキャッシュせず次回再試行
      cache[key] = {"score": score, "reason": reason}
      save_cache(cache)
      time.sleep(1)

    # 反応確認型: スコアの根拠は反応日(a_idx)の大引けまでの情報なので、
    # 買いはその翌営業日の始値(ENTRY_LAG日後)にして先読みを排除する
    e_idx = a_idx + ENTRY_LAG
    if e_idx >= len(df):
      continue  # まだエントリー日が来ていない
    entry_price = df.loc[e_idx, "Open"]
    entry_dt = df.loc[e_idx, "Date"]
    ev = {
        "ticker": ticker,
        "earnings_date": e_str,
        "entry_date": entry_dt,
        "score": score,
        "chg_pct": round(chg, 2),
        "vol_ratio": round(vol_ratio, 2),
        "gap_pct": round(
            (entry_price / p_after - 1) * 100.0 if p_after > 0 else 0, 2
        ),
    }
    # 保有日数ごとのリターン（翌々日Open買 → hold_days日目Close売）と日経平均の同期間
    for h in HOLD_DAYS_LIST:
      x_idx = e_idx + (h - 1)
      if x_idx >= len(df):
        ev[f"ret_{h}"] = None
        ev[f"mae_{h}"] = None
        ev[f"bench_{h}"] = None
        continue
      hold = df.loc[e_idx:x_idx]
      ev[f"ret_{h}"] = (df.loc[x_idx, "Close"] - entry_price) / entry_price * 100.0
      ev[f"mae_{h}"] = (hold["Low"].min() - entry_price) / entry_price * 100.0
      ev[f"bench_{h}"] = bench_return(bench, entry_dt, df.loc[x_idx, "Date"])
    events.append(ev)
  return events


def bench_return(bench, entry_dt, exit_dt):
  """日経平均の同期間リターン(始値→終値)。データがなければ None"""
  if bench is None or entry_dt not in bench.index or exit_dt not in bench.index:
    return None
  o = bench.loc[entry_dt, "Open"]
  return (bench.loc[exit_dt, "Close"] - o) / o * 100.0 if o > 0 else None


# ------------------------------------------------------------------------------
# 2. グリッド集計
# ------------------------------------------------------------------------------
def run_grid(df_ev: pd.DataFrame) -> pd.DataFrame:
  rows = []
  now = df_ev["entry_date"].max()
  for years, thr, h in itertools.product(YEARS_LIST, THRESHOLDS, HOLD_DAYS_LIST):
    cutoff = now - pd.DateOffset(years=years)
    sub = df_ev[(df_ev["entry_date"] >= cutoff) & (df_ev["score"] >= thr)]
    sub = sub[sub[f"ret_{h}"].notna()]
    n = len(sub)
    if n == 0:
      rows.append({"years": years, "threshold": thr, "hold_days": h, "trades": 0})
      continue
    r = sub[f"ret_{h}"]
    excess = (r - sub[f"bench_{h}"]).dropna()
    rows.append({
        "years": years,
        "threshold": thr,
        "hold_days": h,
        "trades": n,
        "win_rate_pct": round((r > 0).mean() * 100, 1),
        "avg_ret_pct": round(r.mean(), 2),
        "avg_excess_pct": round(excess.mean(), 2) if len(excess) else None,
        "median_ret_pct": round(r.median(), 2),
        "worst_ret_pct": round(r.min(), 2),
        "worst_mae_pct": round(sub[f"mae_{h}"].min(), 2),
    })
  return pd.DataFrame(rows)


def main():
  tickers = load_tickers()
  cache = load_cache()
  print(f"グリッド探索開始: {len(tickers)}銘柄 / モデル [{LLM_PROVIDER}/{ACTIVE_MODEL}]")
  print(f"キャッシュ済み: {len(cache)}件\n")

  # 比較対象: 日経平均の日足
  bench = yf.Ticker("^N225").history(period=f"{MAX_YEARS}y").reset_index()
  bench["Date"] = pd.to_datetime(bench["Date"]).dt.tz_localize(None)
  bench = bench.set_index("Date") if not bench.empty else None

  all_events = []
  for i, t in enumerate(tickers, 1):
    print(f"[{i}/{len(tickers)}] {t}")
    evs = collect_events(t, cache, bench)
    all_events.extend(evs)
    print(f"  └─ 決算イベント {len(evs)}件")

  if not all_events:
    print("イベントがありません。")
    return

  # 同一銘柄・同一エントリー日の重複イベントを除去
  df_ev = pd.DataFrame(all_events).drop_duplicates(["ticker", "entry_date"])
  df_ev.to_csv(EVENTS_FILE, index=False, encoding="utf-8-sig")

  grid = run_grid(df_ev)
  grid.to_csv(GRID_RESULT_FILE, index=False, encoding="utf-8-sig")

  ranked = grid[
      (grid["trades"] >= MIN_TRADES) & (grid["threshold"] > 0)
  ].sort_values("avg_excess_pct", ascending=False)
  print("\n=== 全決算を買う場合（threshold=0・比較対象）===")
  print(grid[grid["threshold"] == 0].to_string(index=False))
  print(f"\n=== 日経平均超過リターン上位（トレード数 >= {MIN_TRADES}）===")
  if ranked.empty:
    print("該当なし。MIN_TRADES を下げてください。")
  else:
    print(ranked.head(15).to_string(index=False))
  print(f"\n全組合せ: {GRID_RESULT_FILE} / イベント明細: {EVENTS_FILE}")


if __name__ == "__main__":
  main()
