# 好決算後の翌日寄り付き買い（決算モメンタム）バックテスト実行スクリプト (Groq LLM限定自動検出版)

import json
import os
import time
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------------------
# 0. 環境変数の自動ロード (.env ファイルからの読み込み)
# ------------------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------------------
# 1. 設定パラメータ
# ------------------------------------------------------------------------------
TICKER_SYMBOL = "6758.T"  # テスト対象銘柄（例: ソニーグループ）
BACKTEST_YEARS = 5  # 過去何年分を対象にするか
SCORE_THRESHOLD = 80  # エントリー対象とするAI好決算スコアの閾値
HOLD_DAYS = 5  # 保有日数（例: 翌日Open買エントリー後、N日後のCloseで売却）

# Groq API クライアントの初期化
groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
  print("【エラー】.env ファイルに GROQ_API_KEY が設定されていません。")
  exit(1)

client = Groq(api_key=groq_key)


# ------------------------------------------------------------------------------
# テキスト生成（LLM）専用モデルの自動検出
# ------------------------------------------------------------------------------
def get_working_groq_llm_model() -> str:
  """Groqからチャット生成(LLM)に対応したモデルのみをフィルターして自動取得する"""
  try:
    models_list = client.models.list()
    available_models = [m.id for m in models_list.data]

    # 音声モデル(whisper)や判定モデル(guard)を徹底除外
    llm_models = [
        m
        for m in available_models
        if not any(
            invalid in m.lower()
            for invalid in ["whisper", "guard", "embedding", "vision"]
        )
    ]

    # 優先的に使いたいLLMモデルのキーワード
    preferred_keywords = [
        "llama-3.3-70b",
        "llama-3.1-8b",
        "llama3-70b",
        "llama3-8b",
        "mixtral",
        "gemma",
    ]

    for pref in preferred_keywords:
      for model_id in llm_models:
        if pref in model_id:
          print(f"★ 使用する Groq LLMモデルを確定しました: {model_id}")
          return model_id

    if llm_models:
      print(f"★ 使用する Groq LLMモデル: {llm_models[0]}")
      return llm_models[0]

  except Exception as e:
    print(f"モデル一覧の取得に失敗しました: {e}")

  # 万が一のフォールバック
  return "llama-3.1-8b-instant"


# 起動時に一度だけテキスト生成モデルを取得
ACTIVE_MODEL = get_working_groq_llm_model()


# ------------------------------------------------------------------------------
# 2. AI（Groq API）による決算テキストのスコアリング関数
# ------------------------------------------------------------------------------
def evaluate_earnings_with_groq(
    ticker: str, earnings_date_str: str, earnings_text: str
) -> int:
  """Groq に決算評価を行わせる関数"""
  system_prompt = (
      "あなたは優秀な株式アナリストです。指定されたフォーマットのJSONのみを絶対に出力してください。"
  )
  user_prompt = f"""
以下は、銘柄コード {ticker} の {earnings_date_str} における決算発表データです。
内容（通期進捗率、業績サプライズ、上方修正の有無、今後の見通し）を分析し、
「翌営業日の株価が大幅高・モメンタムを形成する可能性」を0〜100点の数値で評価してください。

【出力フォーマット】
{{"score": 数値(0-100), "reason": "評価理由（簡潔に）"}}

【決算テキスト】
{earnings_text}
"""
  try:
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=ACTIVE_MODEL,
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content.strip()

    # Markdown記法のクレンジング
    cleaned_text = raw_text
    if cleaned_text.startswith("```"):
      cleaned_text = cleaned_text.split("```")[1]
      if cleaned_text.startswith("json"):
        cleaned_text = cleaned_text[4:]
    cleaned_text = cleaned_text.strip()

    res_json = json.loads(cleaned_text)
    return int(res_json.get("score", 0))

  except Exception as e:
    print(f"[{earnings_date_str}] Groq API 評価エラー ({ACTIVE_MODEL}): {e}")
    return 0


# ------------------------------------------------------------------------------
# 3. バックテスト実行エンジン
# ------------------------------------------------------------------------------
def run_earnings_backtest(
    ticker: str, years=2, score_threshold=80, hold_days=3
):
  print(
      f"=== {ticker} のバックテストを開始します（対象期間: 過去{years}年 / Groq"
      f" [{ACTIVE_MODEL}] 使用） ==="
  )

  stock = yf.Ticker(ticker)

  # 日足データの取得
  df_price = stock.history(period=f"{years}y")
  if df_price.empty:
    print("株価データの取得に失敗しました。")
    return

  df_price = df_price.reset_index()
  df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.tz_localize(None)
  df_price = df_price.sort_values("Date").reset_index(drop=True)

  # 過去の決算日の取得
  try:
    earnings_dates = stock.earnings_dates
    if earnings_dates is None or earnings_dates.empty:
      print("決算日データの取得に失敗しました。")
      return
  except Exception as e:
    print(f"決算日データ取得エラー: {e}")
    return

  # 未来の予定日を除外 & タイムゾーン削除
  earnings_dates = earnings_dates.reset_index()
  earnings_dates["Earnings Date"] = pd.to_datetime(
      earnings_dates["Earnings Date"]
  ).dt.tz_localize(None)
  past_earnings = earnings_dates[
      earnings_dates["Earnings Date"] <= df_price["Date"].max()
  ].copy()

  trades = []

  for idx, row in past_earnings.iterrows():
    e_date = row["Earnings Date"].normalize()
    e_date_str = e_date.strftime("%Y-%m-%d")

    dummy_earnings_text = (
        f"第{idx+1}四半期決算短信：営業利益 前年同期比 +25%増益。通期予想を上方修正。"
    )

    # 1. Groq APIによる評価
    score = evaluate_earnings_with_groq(
        ticker, e_date_str, dummy_earnings_text
    )
    print(f"決算日: {e_date_str} | 好決算スコア: {score}点")

    if score < score_threshold:
      print(f" -> スコアが閾値({score_threshold}点)未満のためエントリー見送り")
      continue

    # 2. 翌営業日（寄付き）のエントリー位置特定
    future_prices = df_price[df_price["Date"] > e_date]
    if len(future_prices) < hold_days:
      print(" -> エントリー後の株価データが不十分なためスキップ")
      continue

    entry_idx = future_prices.index[0]
    exit_idx = entry_idx + (hold_days - 1)

    entry_date = df_price.loc[entry_idx, "Date"].strftime("%Y-%m-%d")
    exit_date = df_price.loc[exit_idx, "Date"].strftime("%Y-%m-%d")

    entry_price = df_price.loc[entry_idx, "Open"]  # 翌日寄り付き買い
    exit_price = df_price.loc[exit_idx, "Close"]  # N日後大引け売り

    return_pct = ((exit_price - entry_price) / entry_price) * 100

    trades.append({
        "earnings_date": e_date_str,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "score": score,
        "entry_price": round(entry_price, 1),
        "exit_price": round(exit_price, 1),
        "return_pct": round(return_pct, 2),
    })

    time.sleep(1)

  # 3. 集計結果のレポート出力
  results_df = pd.DataFrame(trades)
  print("\n==================================================")
  print(f"   【バックテスト最終結果】 銘柄: {ticker}")
  print("==================================================")

  if results_df.empty:
    print("条件を満たすトレードは発生しませんでした。")
  else:
    win_trades = results_df[results_df["return_pct"] > 0]
    win_rate = (len(win_trades) / len(results_df)) * 100
    avg_return = results_df["return_pct"].mean()
    max_return = results_df["return_pct"].max()
    min_return = results_df["return_pct"].min()

    print(f"総トレード数 : {len(results_df)} 回")
    print(f"勝率         : {win_rate:.1f} %")
    print(f"平均損益率   : {avg_return:+.2f} %")
    print(f"最大利益率   : {max_return:+.2f} %")
    print(f"最大損失率   : {min_return:+.2f} %")
    print("\n--- 詳細トレードログ ---")
    print(results_df.to_string(index=False))

  return results_df


# ------------------------------------------------------------------------------
# 4. 実行エントリーポイント
# ------------------------------------------------------------------------------
if __name__ == "__main__":
  run_earnings_backtest(
      ticker=TICKER_SYMBOL,
      years=BACKTEST_YEARS,
      score_threshold=SCORE_THRESHOLD,
      hold_days=HOLD_DAYS,
  )