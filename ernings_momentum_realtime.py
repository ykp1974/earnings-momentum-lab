# 当日決算銘柄 リアルタイム自動判定 & S株買い候補抽出スクリプト
# (TDnet リアルタイム取得 ➔ 過去日付・銘柄指定フォールバック ➔ LLM モメンタム判定)

from datetime import datetime
import json
import os
import re
import sys
import time
import urllib.request
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------------------
# 0. 環境変数のロード & 設定パラメータの取得
# ------------------------------------------------------------------------------
load_dotenv()

SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "80"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
TICKER_LIST_FILE = os.getenv("TICKER_LIST_FILE", "ticker_list.txt")

# 日付指定パラメータ (.env から取得。未指定・空欄時は本日)
TARGET_DATE_ENV = os.getenv("TARGET_DATE", "").strip()

# 直接テスト用銘柄指定 (例: "3697,7730,7818" または空欄)
TARGET_TICKERS_ENV = os.getenv("TARGET_TICKERS", "").strip()


# ------------------------------------------------------------------------------
# 1. LLM クライアントのセットアップ & モデル検証
# ------------------------------------------------------------------------------
def create_llm_client():
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
          return model_id

    if llm_models:
      return llm_models[0]
  except Exception as e:
    print(f"モデル検証エラー: {e}")

  return "llama-3.1-8b-instant"


ACTIVE_MODEL = get_active_model_name(LLM_MODEL)


# ------------------------------------------------------------------------------
# 2. TDnet または yfinance からの決算データ取得 (指定銘柄・過去フォールバック対応)
# ------------------------------------------------------------------------------
def fetch_tdnet_earnings(target_date_str: str) -> tuple[list, bool]:
  """TDnet速報/アーカイブから指定日の開示を取得。404等の場合はyfinance等からの代替フラグを返す"""
  today_str = datetime.now().strftime("%Y-%m-%d")

  if not target_date_str or target_date_str == today_str:
    tdnet_url = "https://www.release.tdnet.info/inbs/I_main_00.html"
    display_date = f"{today_str} (本日)"
  else:
    date_clean = target_date_str.replace("-", "").replace("/", "")
    tdnet_url = (
        f"https://www.release.tdnet.info/inbs/I_main_00_{date_clean}.html"
    )
    display_date = target_date_str

  print(f"[1/3] TDnet から {display_date} の開示情報を取得中...")

  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }
  req = urllib.request.Request(tdnet_url, headers=headers)

  try:
    with urllib.request.urlopen(req) as response:
      html = response.read().decode("utf-8", errors="ignore")
      soup = BeautifulSoup(html, "html.parser")
      disclosure_list = []

      rows = soup.find_all("tr")
      for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 5:
          pub_time = cols[0].text.strip()
          code_str = cols[1].text.strip()
          company_name = cols[2].text.strip()
          title = cols[3].text.strip()

          code_match = re.search(r"\d{4}", code_str)
          if not code_match:
            continue
          ticker = f"{code_match.group(0)}.T"

          keywords = [
              "決算短信",
              "業績予想",
              "上方修正",
              "通期業績",
              "四半期",
              "剰余金の配当",
              "本決算",
          ]
          if any(kw in title for kw in keywords):
            disclosure_list.append({
                "ticker": ticker,
                "code": code_match.group(0),
                "name": company_name,
                "pub_time": pub_time,
                "title": title,
            })

      print(
          f"  └─ 対象日の決算・業績関連開示: 全 {len(disclosure_list)} 件検出\n"
      )
      return disclosure_list, False

  except Exception as e:
    print(
        f"  ⚠️ TDnet アーカイブ未存在 ({e})。指定日 ({target_date_str}) の決算シミュレーションモードへ切替ます..."
    )
    return fetch_fallback_earnings(target_date_str), True


def fetch_fallback_earnings(target_date_str: str) -> list:
  """指定銘柄 (TARGET_TICKERS) または ticker_list.txt から過去指定日に決算だった銘柄を検索"""
  tickers = []

  # 1. TARGET_TICKERS が設定されている場合は優先使用
  if TARGET_TICKERS_ENV:
    raw_codes = TARGET_TICKERS_ENV.split(",")
    for code in raw_codes:
      c = code.strip()
      if c:
        tickers.append(c if c.endswith(".T") else f"{c}.T")
  elif os.path.exists(TICKER_LIST_FILE):
    with open(TICKER_LIST_FILE, "r", encoding="utf-8") as f:
      for line in f:
        code = line.strip()
        if code and not code.startswith("#"):
          tickers.append(code if code.endswith(".T") else f"{code}.T")

  fallback_list = []
  target_dt = pd.to_datetime(target_date_str).normalize()

  print(f"  └─ 対象候補銘柄 ({len(tickers)} 銘柄) をスキャン中...")

  for ticker in tickers:
    try:
      stock = yf.Ticker(ticker)
      company_name = (
          stock.info.get("shortName") or stock.info.get("longName") or ticker
      )
      ed = stock.earnings_dates
      if ed is None or ed.empty:
        # earnings_dates が引けない場合は直接検証登録
        fallback_list.append({
            "ticker": ticker,
            "code": ticker.replace(".T", ""),
            "name": company_name,
            "pub_time": "15:00",
            "title": f"本決算/四半期決算短信（{target_date_str} 発表データ）",
        })
        continue

      ed = ed.reset_index()
      ed["Earnings Date"] = pd.to_datetime(ed["Earnings Date"]).dt.tz_localize(
          None
      )

      matched = ed[ed["Earnings Date"].dt.normalize() == target_dt]
      if not matched.empty or TARGET_TICKERS_ENV:
        fallback_list.append({
            "ticker": ticker,
            "code": ticker.replace(".T", ""),
            "name": company_name,
            "pub_time": "15:00",
            "title": f"本決算/四半期決算短信（{target_date_str} 発表データ）",
        })
    except Exception:
      continue

  print(
      f"  └─ 抽出完了: {target_date_str} 対象決算銘柄 {len(fallback_list)} 件検出\n"
  )
  return fallback_list


# ------------------------------------------------------------------------------
# 3. AI（LLM）リアルタイム/過去検証 共通決算解析関数
# ------------------------------------------------------------------------------
def evaluate_realtime_earnings(
    ticker: str, company_name: str, title: str, eval_date_str: str
) -> tuple[int, str]:
  system_prompt = (
      "あなたは優秀な株式アナリストです。指定されたフォーマットのJSONのみを絶対に出力してください。"
  )
  user_prompt = f"""
以下は、{eval_date_str} の大引け後に適時開示（TDnet）にて発表された生の決算関連データです。
「翌営業日以降の株価が大幅高・強力なモメンタムを形成する可能性」を0〜100点の数値で評価してください。

【評価の視点】
・「決算短信」「本決算」かつ通期業績の上方修正や増益、増配などの明確なポジティブサプライズが含まれるタイトル ➔ 80点以上の高得点
・「下方修正」「赤字」などネガティブ、または軽微な修正・通常の開示 ➔ 60点以下の低得点

【出力フォーマット】
{{"score": 数値(0-100), "reason": "評価理由（簡潔に1文で）"}}

【対象銘柄】: {ticker} ({company_name})
【開示タイトル】: {title}
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
    print(f"  [{ticker}] LLM 評価エラー: {e}")
    return 0, f"エラー: {e}"


# ------------------------------------------------------------------------------
# 4. メイン実行パイプライン
# ------------------------------------------------------------------------------
def run_realtime_checker():
  eval_date = (
      TARGET_DATE_ENV if TARGET_DATE_ENV else datetime.now().strftime("%Y-%m-%d")
  )

  print(
      f"\n================================================================================"
  )
  print(f"  【StkLab.】決算判定 & S株買い候補抽出 (判定基準日: {eval_date})")
  print(
      f"  使用 LLM エンジン: [{LLM_PROVIDER.upper()}] / モデル: [{ACTIVE_MODEL}]"
  )
  print(
      f"================================================================================\n"
  )

  disclosures, is_fallback = fetch_tdnet_earnings(eval_date)
  if not disclosures:
    print(
        f"対象日 ({eval_date}) に該当する決算データが見つかりませんでした。"
    )
    return

  print(f"[2/3] AIによる決算データのモメンタム判定中...")

  s_stock_candidates = []

  for i, disc in enumerate(disclosures, 1):
    ticker = disc["ticker"]
    name = disc["name"]
    title = disc["title"]
    pub_time = disc["pub_time"]

    score, reason = evaluate_realtime_earnings(ticker, name, title, eval_date)

    try:
      stock = yf.Ticker(ticker)
      hist = stock.history(period="1mo")
      hist = hist.reset_index()
      hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)

      target_dt = pd.to_datetime(eval_date)
      match_price = hist[hist["Date"] <= target_dt]

      current_price = (
          round(match_price["Close"].iloc[-1], 1)
          if not match_price.empty
          else 0.0
      )
    except Exception:
      current_price = 0.0

    status_icon = "★【S株推奨】" if score >= SCORE_THRESHOLD else "  [見送り]"
    print(
        f"  [{i}/{len(disclosures)}] {disc['code']} {name} | スコア: {score}点"
        f" {status_icon}"
    )
    print(f"      ├ 開示: {title}")
    print(f"      └ 理由: {reason}")

    if score >= SCORE_THRESHOLD:
      s_stock_candidates.append({
          "date": eval_date,
          "pub_time": pub_time,
          "ticker": ticker,
          "code": disc["code"],
          "name": name,
          "score": score,
          "reason": reason,
          "closing_price": current_price,
          "title": title,
          "llm_model": ACTIVE_MODEL,
      })

    time.sleep(1)

  print(
      f"\n================================================================================"
  )
  print(
      f"  【判定完了】{eval_date} の S株（1株）買い候補銘柄リスト (閾値:"
      f" {SCORE_THRESHOLD}点以上)"
  )
  print(
      f"================================================================================"
  )

  if not s_stock_candidates:
    print(
        f"{eval_date} にて AI判定基準（{SCORE_THRESHOLD}点以上）を満たすS株買い候補銘柄はありませんでした。"
    )
    return

  df_candidates = pd.DataFrame(s_stock_candidates)

  for idx, row in df_candidates.iterrows():
    print(f"\n【候補 {idx+1}】 {row['code']} : {row['name']}")
    print(f"  ・AI判定スコア  : {row['score']} 点")
    print(f"  ・発表日終値    : {row['closing_price']} 円 (S株想定金額)")
    print(f"  ・開示タイトル  : {row['title']}")
    print(f"  ・AI推奨理由    : {row['reason']}")

  output_csv = f"realtime_s_stock_candidates_{eval_date.replace('-', '')}.csv"
  df_candidates.to_csv(output_csv, index=False, encoding="utf-8-sig")
  print(
      f"\n--------------------------------------------------------------------------------"
  )
  print(f"★ S株発注候補リストを '{output_csv}' に出力保存しました。")
  print(
      "--------------------------------------------------------------------------------\n"
  )


if __name__ == "__main__":
  run_realtime_checker()