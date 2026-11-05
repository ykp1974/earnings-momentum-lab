# LLM を使わないルール検証: 反応日の株価変化率・出来高倍率の条件をずらして成績を比べる
# grid_events.csv（grid_backtest.py の出力）だけを入力にするので API は不要。

import itertools
import sys

import pandas as pd

# 使い方: python rule_grid.py [イベントCSV] [結果CSV]
EVENTS_FILE = sys.argv[1] if len(sys.argv) > 1 else "grid_events.csv"
RESULT_FILE = sys.argv[2] if len(sys.argv) > 2 else "rule_grid_result.csv"

CHG_LIST = [0, 2, 3, 5, 7, 10]  # 反応日の株価変化率(%)の下限
VOL_LIST = [0, 1.5, 2, 3]  # 出来高倍率の下限
HOLD_DAYS_LIST = [3, 5, 10]
YEARS_LIST = [3, 5]
MIN_TRADES = 30


def summarize(sub: pd.DataFrame, h: int) -> dict:
  sub = sub[sub[f"ret_{h}"].notna()]
  r = sub[f"ret_{h}"]
  ex = (r - sub[f"bench_{h}"]).dropna()
  if len(r) == 0:
    return {"trades": 0}
  return {
      "trades": len(r),
      "win_rate_pct": round((r > 0).mean() * 100, 1),
      "avg_ret_pct": round(r.mean(), 2),
      "avg_excess_pct": round(ex.mean(), 2) if len(ex) else None,
      "excess_se": round(ex.std() / len(ex) ** 0.5, 2) if len(ex) > 1 else None,
  }


def main():
  df = pd.read_csv(EVENTS_FILE, encoding="utf-8-sig", parse_dates=["entry_date"])
  now = df["entry_date"].max()
  mid = df["entry_date"].min() + (now - df["entry_date"].min()) / 2

  rows = []
  for years, chg, vol, h in itertools.product(
      YEARS_LIST, CHG_LIST, VOL_LIST, HOLD_DAYS_LIST
  ):
    base = df[df["entry_date"] >= now - pd.DateOffset(years=years)]
    sub = base[(base["chg_pct"] >= chg) & (base["vol_ratio"] >= vol)]
    row = {"years": years, "min_chg": chg, "min_vol": vol, "hold_days": h}
    row.update(summarize(sub, h))
    # 前半・後半に分けて再現性を見る（過剰最適化の確認）
    for name, part in (
        ("first", sub[sub["entry_date"] < mid]),
        ("second", sub[sub["entry_date"] >= mid]),
    ):
      s = summarize(part, h)
      row[f"excess_{name}"] = s.get("avg_excess_pct")
      row[f"n_{name}"] = s.get("trades", 0)
    rows.append(row)

  res = pd.DataFrame(rows)
  res.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

  ok = res[res["trades"] >= MIN_TRADES].sort_values(
      "avg_excess_pct", ascending=False
  )
  pd.set_option("display.width", 220)
  print(f"組合せ数: {len(res)} / トレード数 {MIN_TRADES} 以上: {len(ok)}")
  print("\n=== 超過リターン上位 ===")
  print(ok.head(15).to_string(index=False))
  print("\n=== 前半・後半とも超過リターンがプラスの組合せ ===")
  both = ok[(ok["excess_first"] > 0) & (ok["excess_second"] > 0)]
  print(both.head(15).to_string(index=False) if len(both) else "なし")
  print(f"\n全結果: {RESULT_FILE}")


if __name__ == "__main__":
  main()
