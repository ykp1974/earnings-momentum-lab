# 好決算後の翌日寄り付き買い（決算モメンタム）マルチLLM対応バックテストスクリプト (本物決算ファクトデータ生成版)

import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------------------
# 0. 環境変数のロード & 設定パラメータの取得
# ------------------------------------------------------------------------------
load_dotenv()

TICKER_LIST_FILE = os.getenv("TICKER_LIST_FILE", "ticker_list.txt")
BACKTEST_YEARS = int(os.getenv("BACKTEST_YEARS", "5"))
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "80"))
HOLD_DAYS = int(os.getenv("HOLD_DAYS", "5"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")


# ------------------------------------------------------------------------------
# 1. LLM クライアント（OpenAI 互換）の動的セットアップ & モデル検証
# ------------------------------------------------------------------------------
def create_llm_client():
  """.env の設定に応じて適切な API クライアントとエンドポイントを生成する"""
  if LLM_PROVIDER == "groq":
    api_key = os.getenv("GROQ_API_KEY")
    base_url = "https://api.groq.com/openai/v1"
  elif LLM_PROVIDER == "siliconflow":
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = "https://api.siliconflow.cn/v1"
  elif LLM_PROVIDER == "deepseek":
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = "https://api.deepseek.com"
  else:
    raise ValueError(f"未対応の LLM プロバイダーです: {LLM_PROVIDER}")

  if not api_key:
    print(
        f"【エラー】 .env に {LLM_PROVIDER.upper()}_API_KEY"
        " が設定されていません。"
    )
    exit(1)

  return OpenAI(api_key=api_key, base_url=base_url)


client = create_llm_client()


def get_active_model_name(requested_model: str) -> str:
  """指定されたモデルが利用可能か検証し、エラー時は利用可能なテキスト生成モデルを自動選択する"""
  if LLM_PROVIDER != "groq":
    return requested_model

  try:
    models_list = client.models.list()
    available_models = [m.id for m in models_list.data]

    if requested_model in available_models:
      return requested_model

    llm_models = [
        m
        for m in available_models
        if not any(
            invalid in m.lower()
            for invalid in [
                "whisper",
                "guard",
                "embedding",
                "vision",
                "canopylabs",
                "orpheus",
                "tts",
            ]
        )
    ]

    preferred_keywords = [
        "llama-3.3",
        "llama-3.1-8b",
        "llama-3.1-70b",
        "llama3-70b",
        "llama3-8b",
        "mixtral",
    ]
    for pref in preferred_keywords:
      for model_id in llm_models:
        if pref in model_id:
          print(
              f"⚠️ 指定モデル ({requested_model}) が見つからないため、最適モデル"
              f" ({model_id}) に自動切替しました。"
          )
          return model_id

    if llm_models:
      return llm_models[0]
  except Exception as e:
    print(f"モデル一覧検証エラー: {e}")

  return "llama-3.1-8b-instant"


ACTIVE_MODEL = get_active_model_name(LLM_MODEL)


# ------------------------------------------------------------------------------
# 2. AI 決算評価関数 (スコア & 理由の取得)
# ------------------------------------------------------------------------------
def evaluate_earnings_with_llm(
    ticker: str, earnings_date_str: str, earnings_text: str
) -> tuple[int, str]:
  """LLM経由で決算テキストのスコアリングと評価理由を返却する tuple(score, reason)"""
  system_prompt = (
      "あなたは優秀な株式アナリストです。指定されたフォーマットのJSONのみを絶対に出力してください。"
  )
  user_prompt = f"""
以下は、銘柄コード {ticker} の {earnings_date_str} における決算発表データ・初期反応サマリーです。
「翌営業日以降の株価が大幅高・強力なモメンタムを形成する可能性」を0〜100点の数値で評価してください。

【評価基準の目安】
・決算発表直後に株価が大幅上昇（+5%以上）かつ出来高が急増（2倍以上）している場合 ➔ 80点以上の高スコアを付与
・反応が薄い、または下落・出来高不調の場合 ➔ 60点以下の低スコアを付与

【出力フォーマット】
{{"score": 数値(0-100), "reason": "評価理由（簡潔に1文で）"}}

【決算サマリーデータ】
{earnings_text}
"""
  try:
    kwargs = {
        "model": ACTIVE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }

    if LLM_PROVIDER in ["groq", "deepseek"]:
      kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    raw_text = response.choices[0].message.content.strip()

    cleaned_text = raw_text
    if cleaned_text.startswith("```"):
      cleaned_text = cleaned_text.split("```")[1]
      if cleaned_text.startswith("json"):
        cleaned_text = cleaned_text[4:]
    cleaned_text = cleaned_text.strip()

    res_json = json.loads(cleaned_text)
    score = int(res_json.get("score", 0))
    reason = str(res_json.get("reason", "理由未記述"))
    return score, reason

  except Exception as e:
    print(
        f"  [{ticker} {earnings_date_str}] LLM 評価エラー"
        f" ({LLM_PROVIDER}/{ACTIVE_MODEL}): {e}"
    )
    return 0, f"エラー発生: {e}"


# ------------------------------------------------------------------------------
# 3. 単一銘柄のバックテスト処理 (過去の実像決算ファクト生成ロジック追加)
# ------------------------------------------------------------------------------
def backtest_single_ticker(
    ticker: str, years: int, score_threshold: int, hold_days: int
) -> list:
  stock = yf.Ticker(ticker)

  try:
    company_name = (
        stock.info.get("shortName") or stock.info.get("longName") or ticker
    )
  except Exception:
    company_name = ticker

  df_price = stock.history(period=f"{years}y")
  if df_price.empty:
    print(f"  └─ ⚠️ 株価データが取得できませんでした。スキップします。")
    return []

  df_price = df_price.reset_index()
  df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.tz_localize(None)
  df_price = df_price.sort_values("Date").reset_index(drop=True)

  try:
    earnings_dates = stock.earnings_dates
    if earnings_dates is None or earnings_dates.empty:
      print(f"  └─ ⚠️ 決算日データが存在しません。スキップします。")
      return []
  except Exception:
    print(
        f"  └─ ⚠️ 決算日データの取得に失敗しました。スキップします。"
    )
    return []

  earnings_dates = earnings_dates.reset_index()
  earnings_dates["Earnings Date"] = pd.to_datetime(
      earnings_dates["Earnings Date"]
  ).dt.tz_localize(None)
  past_earnings = earnings_dates[
      earnings_dates["Earnings Date"] <= df_price["Date"].max()
  ].copy()

  ticker_trades = []

  for idx, row in past_earnings.iterrows():
    e_date = row["Earnings Date"].normalize()
    e_date_str = e_date.strftime("%Y-%m-%d")

    # --- 【改善】過去の実際の株価・出来高から本物の「決算反応サマリー」を生成 ---
    prev_prices = df_price[df_price["Date"] <= e_date]
    future_prices = df_price[df_price["Date"] > e_date]

    if prev_prices.empty or len(future_prices) < hold_days:
      continue

    # 決算直前の価格・出来高と、決算直後の価格・出来高を取得
    last_before_idx = prev_prices.index[-1]
    first_after_idx = future_prices.index[0]

    price_before = df_price.loc[last_before_idx, "Close"]
    price_after = df_price.loc[first_after_idx, "Close"]
    vol_before = df_price.loc[last_before_idx, "Volume"]
    vol_after = df_price.loc[first_after_idx, "Volume"]

    # 変化率の計算
    price_change_pct = (
        ((price_after - price_before) / price_before) * 100.0
        if price_before > 0
        else 0
    )
    vol_ratio = (vol_after / vol_before) if vol_before > 0 else 1.0

    # AIに読ませる本物の「決算サマリーデータ」を構築
    earnings_fact_text = (
        f"【決算発表日】: {e_date_str}\n"
        f"【決算直後の市場反応】: 株価変化率 {price_change_pct:+.2f}%, 出来高変動"
        f" {vol_ratio:.1f}倍\n"
        f"【株価動向サマリー】: 発表直前の株価 {price_before:.1f}円 ➔ 発表直後の株価"
        f" {price_after:.1f}円"
    )

    score, reason = evaluate_earnings_with_llm(
        ticker, e_date_str, earnings_fact_text
    )

    if score < score_threshold:
      continue

    entry_idx = first_after_idx
    exit_idx = entry_idx + (hold_days - 1)

    hold_period_df = df_price.loc[entry_idx:exit_idx]

    entry_date = df_price.loc[entry_idx, "Date"].strftime("%Y-%m-%d")
    exit_date = df_price.loc[exit_idx, "Date"].strftime("%Y-%m-%d")

    entry_price = df_price.loc[entry_idx, "Open"]
    exit_price = df_price.loc[exit_idx, "Close"]

    return_pct = ((exit_price - entry_price) / entry_price) * 100.0

    max_high = hold_period_df["High"].max()
    min_low = hold_period_df["Low"].min()
    mfe_pct = ((max_high - entry_price) / entry_price) * 100.0
    mae_pct = ((min_low - entry_price) / entry_price) * 100.0

    trade_record = {
        "ticker": ticker,
        "name": company_name,
        "earnings_date": e_date_str,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "score": score,
        "reason": reason,
        "llm_provider": LLM_PROVIDER,
        "llm_model": ACTIVE_MODEL,
        "entry_price": round(entry_price, 1),
        "exit_price": round(exit_price, 1),
        "return_pct": round(return_pct, 2),
        "mfe_pct": round(mfe_pct, 2),
        "mae_pct": round(mae_pct, 2),
    }
    ticker_trades.append(trade_record)

    print(
        f"  [決算日: {e_date_str}] AIスコア: {score}点 ({reason}) ➔ 翌日Open買"
        f" ({entry_price:.1f}円) ➔ {hold_days}日後Close売 ({exit_price:.1f}円) |"
        f" 損益: {return_pct:+.2f}% (最大含み益: +{mfe_pct:.2f}% / 最大含み損:"
        f" {mae_pct:.2f}%)"
    )

    time.sleep(1)

  return ticker_trades


# ------------------------------------------------------------------------------
# 4. 全銘柄ループ & パフォーマンス集計
# ------------------------------------------------------------------------------
def run_multi_ticker_backtest():
  if not os.path.exists(TICKER_LIST_FILE):
    print(f"【エラー】リストファイル '{TICKER_LIST_FILE}' が見つかりません。")
    return

  tickers = []
  with open(TICKER_LIST_FILE, "r", encoding="utf-8") as f:
    for line in f:
      code = line.strip()
      if code and not code.startswith("#"):
        if not code.endswith(".T"):
          code = f"{code}.T"
        tickers.append(code)

  print(
      f"\n================================================================================"
  )
  print(
      f"  複数銘柄バックテスト開始 (全 {len(tickers)} 銘柄 / 過去 {BACKTEST_YEARS}"
      f" 年 / 保有 {HOLD_DAYS} 日)"
  )
  print(
      f"  使用 LLM エンジン: [{LLM_PROVIDER.upper()}] / モデル: [{ACTIVE_MODEL}]"
  )
  print(
      f"================================================================================\n"
  )

  all_trades = []

  for i, ticker in enumerate(tickers, 1):
    print(
        f"--------------------------------------------------------------------------------"
    )
    print(f" [{i}/{len(tickers)}] {ticker} の検証処理中...")
    print(
        f"--------------------------------------------------------------------------------"
    )

    trades = backtest_single_ticker(
        ticker, BACKTEST_YEARS, SCORE_THRESHOLD, HOLD_DAYS
    )
    if trades:
      all_trades.extend(trades)
      win_count = sum(1 for t in trades if t["return_pct"] > 0)
      win_rate = (win_count / len(trades)) * 100.0
      avg_ret = sum(t["return_pct"] for t in trades) / len(trades)
      print(
          f"  └─ 完了: {len(trades)}戦{win_count}勝{len(trades)-win_count}敗 (勝率"
          f" {win_rate:.1f}% / 平均損益 {avg_ret:+.2f}%)\n"
      )
    else:
      print(
          f"  └─ 条件を満たすトレードシグナルはありませんでした（AIが80点未満と判定）。\n"
      )

  print(
      "\n================================================================================"
  )
  print("   【全銘柄 統合ポートフォリオ パフォーマンスランキング】")
  print(
      "================================================================================"
  )

  if not all_trades:
    print("全体で条件を満たすトレードは1件も発生しませんでした。")
    return

  df_all = pd.DataFrame(all_trades)

  win_trades = df_all[df_all["return_pct"] > 0]
  overall_win_rate = (len(win_trades) / len(df_all)) * 100.0
  overall_avg_return = df_all["return_pct"].mean()
  overall_max_return = df_all["return_pct"].max()
  overall_min_return = df_all["return_pct"].min()

  print(f"総検証銘柄数 : {len(tickers)} 銘柄")
  print(f"総トレード数 : {len(df_all)} 回")
  print(
      f"全体勝率     : {overall_win_rate:.1f} % ({len(win_trades)}勝"
      f" {len(df_all)-len(win_trades)}敗)"
  )
  print(f"全体平均損益 : {overall_avg_return:+.2f} %")
  print(f"最大利益率   : {overall_max_return:+.2f} %")
  print(f"最大損失率   : {overall_min_return:+.2f} %")

  print("\n--- 銘柄別パフォーマンスランキング（平均損益順） ---")
  summary_df = (
      df_all.groupby(["ticker", "name"])
      .agg(
          トレード数=("return_pct", "count"),
          勝率_pct=("return_pct", lambda x: round((x > 0).mean() * 100, 1)),
          平均損益_pct=("return_pct", lambda x: round(x.mean(), 2)),
          最大利幅_pct=("mfe_pct", lambda x: round(x.max(), 2)),
          最大掘幅_pct=("mae_pct", lambda x: round(x.min(), 2)),
      )
      .reset_index()
  )

  summary_df = summary_df.sort_values(
      by="平均損益_pct", ascending=False
  ).reset_index(drop=True)
  summary_df.index += 1

  summary_df.columns = [
      "Ticker",
      "銘柄名",
      "トレード数",
      "勝率(%)",
      "平均損益(%)",
      "最大含み益(%)",
      "最大含み損(%)",
  ]
  print(summary_df.to_string())

  output_filename = "multi_backtest_result.csv"
  df_all.to_csv(output_filename, index=False, encoding="utf-8-sig")
  print(
      f"\n=========================================================================="
  )
  print(
      f"★ 全 {len(df_all)} 件の詳細ログを '{output_filename}' にCSV保存しました。"
  )
  print(
      f"=========================================================================="
  )


if __name__ == "__main__":
  run_multi_ticker_backtest()