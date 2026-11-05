# LLM を使わないイベント収集: 決算ごとの反応日指標とリターンを yfinance だけで集める
# 使い方: python collect_events_nollm.py <銘柄リスト> [出力CSV] [除外リスト]
#   銘柄リスト内のコードは、行内の「8035」「（8035）」「8035.T」のいずれの形式でも拾う
#   除外リスト（既定: ticker_list.txt）に載っている銘柄は検証済みとして飛ばす
# 売買ルールは grid_backtest.py と同じ（反応日の翌営業日の始値で買い、保有日数目の終値で売る）

import re
import sys
import time

import pandas as pd
import yfinance as yf

MAX_YEARS = 5
HOLD_DAYS_LIST = [3, 4, 5, 7, 10]
import os
ENTRY_LAG = int(os.getenv("ENTRY_LAG", "1"))


def read_codes(path: str) -> list[str]:
  codes = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      if line.strip().startswith("#"):
        continue
      codes += re.findall(r"(?<![0-9A-Z])(\d{3}[0-9A-Z])(?![0-9A-Z])", line)
  return list(dict.fromkeys(codes))


def fetch_with_retry(fn, tries=3):
  for k in range(tries):
    try:
      return fn()
    except Exception as e:
      if k == tries - 1:
        print(f"  └─ ⚠️ 取得失敗: {e}")
        return None
      time.sleep(5 * (k + 1))


def bench_return(bench, entry_dt, exit_dt):
  if bench is None or entry_dt not in bench.index or exit_dt not in bench.index:
    return None
  o = bench.loc[entry_dt, "Open"]
  return (bench.loc[exit_dt, "Close"] - o) / o * 100.0 if o > 0 else None


def collect(ticker: str, bench) -> list[dict]:
  stock = yf.Ticker(ticker)
  df = fetch_with_retry(lambda: stock.history(period=f"{MAX_YEARS}y"))
  if df is None or df.empty:
    print("  └─ ⚠️ 株価データなし。スキップ")
    return []
  df = df.reset_index()
  df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
  df = df.sort_values("Date").reset_index(drop=True)

  ed = fetch_with_retry(lambda: stock.earnings_dates)
  if ed is None or ed.empty:
    print("  └─ ⚠️ 決算日データなし。スキップ")
    return []
  ed = ed.reset_index()
  ed["Earnings Date"] = pd.to_datetime(ed["Earnings Date"]).dt.tz_localize(None)
  ed = ed[ed["Earnings Date"] <= df["Date"].max()]

  events = []
  for _, row in ed.iterrows():
    e_date = row["Earnings Date"].normalize()
    before = df[df["Date"] <= e_date]
    after = df[df["Date"] > e_date]
    if before.empty or after.empty:
      continue
    b_idx, a_idx = before.index[-1], after.index[0]
    p_before, p_after = df.loc[b_idx, "Close"], df.loc[a_idx, "Close"]
    v_before, v_after = df.loc[b_idx, "Volume"], df.loc[a_idx, "Volume"]
    e_idx = a_idx + ENTRY_LAG
    if e_idx >= len(df):
      continue
    entry_price = df.loc[e_idx, "Open"]
    entry_dt = df.loc[e_idx, "Date"]
    ev = {
        "ticker": ticker,
        "earnings_date": e_date.strftime("%Y-%m-%d"),
        "entry_date": entry_dt,
        "chg_pct": round((p_after - p_before) / p_before * 100.0, 2) if p_before > 0 else 0,
        "vol_ratio": round(v_after / v_before, 2) if v_before > 0 else 1.0,
        "gap_pct": round((entry_price / p_after - 1) * 100.0, 2) if p_after > 0 else 0,
    }
    for h in HOLD_DAYS_LIST:
      x_idx = e_idx + (h - 1)
      if x_idx >= len(df):
        ev[f"ret_{h}"] = ev[f"mae_{h}"] = ev[f"bench_{h}"] = None
        continue
      hold = df.loc[e_idx:x_idx]
      ev[f"ret_{h}"] = (df.loc[x_idx, "Close"] - entry_price) / entry_price * 100.0
      ev[f"mae_{h}"] = (hold["Low"].min() - entry_price) / entry_price * 100.0
      ev[f"bench_{h}"] = bench_return(bench, entry_dt, df.loc[x_idx, "Date"])
    events.append(ev)
  return events


def main():
  src = sys.argv[1]
  out = sys.argv[2] if len(sys.argv) > 2 else "grid_events_new.csv"
  excl = sys.argv[3] if len(sys.argv) > 3 else "ticker_list.txt"

  done = set(read_codes(excl))
  codes = [c for c in read_codes(src) if c not in done]
  print(f"対象 {len(codes)} 銘柄（検証済み {len(done)} 銘柄を除外）→ {out}")

  bench = yf.Ticker("^N225").history(period=f"{MAX_YEARS}y").reset_index()
  bench["Date"] = pd.to_datetime(bench["Date"]).dt.tz_localize(None)
  bench = bench.set_index("Date")

  all_events = []
  for i, c in enumerate(codes, 1):
    t = f"{c}.T"
    print(f"[{i}/{len(codes)}] {t}", flush=True)
    evs = collect(t, bench)
    print(f"  └─ 決算イベント {len(evs)}件", flush=True)
    all_events += evs
    pd.DataFrame(all_events).to_csv(out, index=False, encoding="utf-8-sig")
    time.sleep(1)

  df = pd.DataFrame(all_events).drop_duplicates(["ticker", "entry_date"])
  df.to_csv(out, index=False, encoding="utf-8-sig")
  print(f"\n完了: {len(df)}件を {out} に保存")


if __name__ == "__main__":
  main()
