# 好決算後の翌日寄り付き買い（決算モメンタム）バックテスト
# ストップ高考慮版: 翌日がストップ高だった場合の最悪執行価格（高値）でのInを検証
#
# 通常ケース    : 翌日の Open でエントリー（既存ロジックと同一）
# ストップ高ケース: 翌日の High でエントリー（ストップ高が外れた瞬間の最悪成立を想定）
#
# ストップ高の判定: 翌日 Open が前日 Close から +STOPHIGH_GAP_PCT% 以上のギャップアップ

import os
import time
import json
import pandas as pd
import yfinance as yf
from groq import Groq
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 0. 初期設定 & 環境変数ロード
# ------------------------------------------------------------------------------
load_dotenv()

TICKER_LIST_FILE   = os.getenv("TICKER_LIST_FILE", "ticker_list.txt")
BACKTEST_YEARS     = int(os.getenv("BACKTEST_YEARS", "5"))
SCORE_THRESHOLD    = int(os.getenv("SCORE_THRESHOLD", "80"))
HOLD_DAYS          = int(os.getenv("HOLD_DAYS", "5"))
STOPHIGH_GAP_PCT   = float(os.getenv("STOPHIGH_GAP_PCT", "20.0"))  # ストップ高とみなすギャップ率(%)

groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
    print("【エラー】.env ファイルに GROQ_API_KEY が設定されていません。")
    exit(1)

client = Groq(api_key=groq_key)

# ------------------------------------------------------------------------------
# 1. Groq LLM モデル自動検出
# ------------------------------------------------------------------------------
def get_working_groq_llm_model() -> str:
    try:
        models_list = client.models.list()
        available_models = [m.id for m in models_list.data]
        llm_models = [
            m for m in available_models
            if not any(k in m.lower() for k in ["whisper", "guard", "embedding", "vision"])
        ]
        for pref in ["llama-3.3-70b", "llama-3.1-8b", "llama3-70b", "llama3-8b", "mixtral", "gpt-oss"]:
            for model_id in llm_models:
                if pref in model_id:
                    print(f"★ 使用する Groq LLMモデルを決定しました: {model_id}")
                    return model_id
        if llm_models:
            return llm_models[0]
    except Exception as e:
        print(f"モデル一覧の取得に失敗: {e}")
    return "llama-3.1-8b-instant"

ACTIVE_MODEL = get_working_groq_llm_model()

# ------------------------------------------------------------------------------
# 2. AI（Groq API）決算評価関数
# ------------------------------------------------------------------------------
def evaluate_earnings_with_groq(ticker: str, earnings_date_str: str, earnings_text: str) -> int:
    system_prompt = "あなたは優秀な株式アナリストです。指定されたフォーマットのJSONのみを絶対に出力してください。"
    user_prompt = f"""
以下は、銘柄コード {ticker} の {earnings_date_str} における決算発表データです。
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
                {"role": "user", "content": user_prompt}
            ],
            model=ACTIVE_MODEL,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()
        res_json = json.loads(raw_text)
        return int(res_json.get("score", 0))
    except Exception as e:
        print(f"  [{ticker} {earnings_date_str}] API評価エラー: {e}")
        return 0

# ------------------------------------------------------------------------------
# 3. ストップ高判定ヘルパー
# ------------------------------------------------------------------------------
def detect_stophigh(open_price: float, prev_close: float, threshold_pct: float) -> bool:
    """前日終値から threshold_pct% 以上ギャップアップしていればストップ高とみなす"""
    if prev_close <= 0:
        return False
    gap = (open_price - prev_close) / prev_close * 100
    return gap >= threshold_pct

# ------------------------------------------------------------------------------
# 4. 単一銘柄バックテスト（ストップ高考慮版）
# ------------------------------------------------------------------------------
def backtest_single_ticker(ticker: str, years: int, score_threshold: int, hold_days: int) -> list:
    stock = yf.Ticker(ticker)

    try:
        company_name = stock.info.get("shortName") or stock.info.get("longName") or ticker
    except Exception:
        company_name = ticker

    df_price = stock.history(period=f"{years}y")
    if df_price.empty:
        print(f"  └─ 株価データが取得できませんでした。スキップします。")
        return []

    df_price = df_price.reset_index()
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.tz_localize(None)
    df_price = df_price.sort_values('Date').reset_index(drop=True)

    try:
        earnings_dates = stock.earnings_dates
        if earnings_dates is None or earnings_dates.empty:
            print(f"  └─ 決算日データが存在しません。スキップします。")
            return []
    except Exception:
        print(f"  └─ 決算日データの取得に失敗しました。スキップします。")
        return []

    earnings_dates = earnings_dates.reset_index()
    earnings_dates['Earnings Date'] = pd.to_datetime(earnings_dates['Earnings Date']).dt.tz_localize(None)
    past_earnings = earnings_dates[earnings_dates['Earnings Date'] <= df_price['Date'].max()].copy()

    ticker_trades = []

    for idx, row in past_earnings.iterrows():
        e_date = row['Earnings Date'].normalize()
        e_date_str = e_date.strftime('%Y-%m-%d')
        dummy_earnings_text = f"第{idx+1}四半期決算短信：営業利益 前年同期比 +25%増益。通期予想を上方修正。"

        score = evaluate_earnings_with_groq(ticker, e_date_str, dummy_earnings_text)

        if score < score_threshold:
            continue

        future_prices = df_price[df_price['Date'] > e_date]
        if len(future_prices) < hold_days:
            continue

        entry_idx = future_prices.index[0]
        exit_idx  = entry_idx + (hold_days - 1)

        entry_date = df_price.loc[entry_idx, 'Date'].strftime('%Y-%m-%d')
        exit_date  = df_price.loc[exit_idx, 'Date'].strftime('%Y-%m-%d')

        entry_open  = df_price.loc[entry_idx, 'Open']
        entry_high  = df_price.loc[entry_idx, 'High']
        exit_price  = df_price.loc[exit_idx, 'Close']

        # 前日終値を取得（先頭行ガード付き）
        prev_close = df_price.loc[entry_idx - 1, 'Close'] if entry_idx > 0 else entry_open
        gap_pct    = (entry_open - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

        # ストップ高判定
        is_stop_high = detect_stophigh(entry_open, prev_close, STOPHIGH_GAP_PCT)

        # ストップ高 → 高値でIn（最悪執行）、通常 → 始値でIn（従来ロジック）
        entry_price = entry_high if is_stop_high else entry_open

        hold_period_df = df_price.loc[entry_idx:exit_idx]
        return_pct = ((exit_price - entry_price) / entry_price) * 100
        max_high   = hold_period_df['High'].max()
        min_low    = hold_period_df['Low'].min()
        mfe_pct    = ((max_high - entry_price) / entry_price) * 100
        mae_pct    = ((min_low  - entry_price) / entry_price) * 100

        stophigh_label = f"[ストップ高 gap={gap_pct:+.1f}% High執行]" if is_stop_high else f"[通常 gap={gap_pct:+.1f}% Open執行]"

        trade_record = {
            'ticker':        ticker,
            'name':          company_name,
            'earnings_date': e_date_str,
            'entry_date':    entry_date,
            'exit_date':     exit_date,
            'score':         score,
            'entry_open':    round(entry_open,  1),
            'entry_high':    round(entry_high,  1),
            'entry_price':   round(entry_price, 1),  # 実際に使用した執行価格
            'exit_price':    round(exit_price,  1),
            'prev_close':    round(prev_close,  1),
            'gap_pct':       round(gap_pct,     2),
            'is_stop_high':  is_stop_high,
            'return_pct':    round(return_pct,  2),
            'mfe_pct':       round(mfe_pct,     2),
            'mae_pct':       round(mae_pct,     2),
        }
        ticker_trades.append(trade_record)

        print(f"[{ticker}:{company_name}] [{e_date_str}] スコア:{score}点 {stophigh_label} "
              f"執行:{entry_price:.1f}円 -> {hold_days}日後Close:{exit_price:.1f}円 | "
              f"損益:{return_pct:+.2f}% (MFE:+{mfe_pct:.2f}% / MAE:{mae_pct:.2f}%)")

        time.sleep(1)

    return ticker_trades

# ------------------------------------------------------------------------------
# 5. 全銘柄ループ & レポート
# ------------------------------------------------------------------------------
def run_multi_ticker_backtest():
    if not os.path.exists(TICKER_LIST_FILE):
        print(f"【エラー】リストファイル '{TICKER_LIST_FILE}' が見つかりません。")
        return

    with open(TICKER_LIST_FILE, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"\n{'='*80}")
    print(f"  ストップ高考慮版 バックテスト開始")
    print(f"  全 {len(tickers)} 銘柄 / 過去 {BACKTEST_YEARS} 年間 / 保有 {HOLD_DAYS} 日")
    print(f"  ストップ高判定閾値: ギャップアップ +{STOPHIGH_GAP_PCT:.0f}% 以上")
    print(f"{'='*80}\n")

    all_trades = []

    for i, ticker in enumerate(tickers, 1):
        print(f"{'-'*80}")
        print(f" [{i}/{len(tickers)}] {ticker} の検証処理中...")
        print(f"{'-'*80}")

        trades = backtest_single_ticker(ticker, BACKTEST_YEARS, SCORE_THRESHOLD, HOLD_DAYS)
        if trades:
            all_trades.extend(trades)
            win_count = sum(1 for t in trades if t['return_pct'] > 0)
            win_rate  = (win_count / len(trades)) * 100
            avg_ret   = sum(t['return_pct'] for t in trades) / len(trades)
            sh_count  = sum(1 for t in trades if t['is_stop_high'])
            print(f"  └─ 完了: {len(trades)}戦{win_count}勝{len(trades)-win_count}敗 "
                  f"(勝率 {win_rate:.1f}% / 平均損益 {avg_ret:+.2f}%) "
                  f"うちストップ高 {sh_count}件\n")
        else:
            print(f"  └─ 条件を満たすシグナルはありませんでした。\n")

    if not all_trades:
        print("全体で条件を満たすトレードは1件も発生しませんでした。")
        return

    df_all = pd.DataFrame(all_trades)

    # --------------------------------------------------------------------------
    # レポート: 全体
    # --------------------------------------------------------------------------
    def _stats(df: pd.DataFrame, label: str):
        if df.empty:
            print(f"  {label}: 該当なし")
            return
        wins     = (df['return_pct'] > 0).sum()
        total    = len(df)
        win_rate = wins / total * 100
        avg_ret  = df['return_pct'].mean()
        max_ret  = df['return_pct'].max()
        min_ret  = df['return_pct'].min()
        print(f"  {label}")
        print(f"    件数     : {total} 件")
        print(f"    勝率     : {win_rate:.1f}% ({wins}勝{total-wins}敗)")
        print(f"    平均損益 : {avg_ret:+.2f}%")
        print(f"    最大利益 : {max_ret:+.2f}%")
        print(f"    最大損失 : {min_ret:+.2f}%")

    print(f"\n{'='*80}")
    print("   【全銘柄 統合ポートフォリオ パフォーマンスレポート】")
    print(f"{'='*80}")

    _stats(df_all, "■ 全トレード（ストップ高含む）")
    print()
    _stats(df_all[~df_all['is_stop_high']], "■ 通常エントリー（Open執行）のみ")
    print()
    _stats(df_all[df_all['is_stop_high']],  "■ ストップ高エントリー（High執行）のみ")

    # --------------------------------------------------------------------------
    # 銘柄別ランキング
    # --------------------------------------------------------------------------
    print(f"\n--- 銘柄別パフォーマンスランキング（平均損益順） ---")
    summary_df = df_all.groupby(['ticker', 'name']).agg(
        トレード数=('return_pct', 'count'),
        ストップ高数=('is_stop_high', 'sum'),
        勝率_pct=('return_pct', lambda x: round((x > 0).mean() * 100, 1)),
        平均損益_pct=('return_pct', lambda x: round(x.mean(), 2)),
        最大含み益_pct=('mfe_pct', lambda x: round(x.max(), 2)),
        最大含み損_pct=('mae_pct', lambda x: round(x.min(), 2))
    ).reset_index()

    summary_df = summary_df.sort_values(by='平均損益_pct', ascending=False).reset_index(drop=True)
    summary_df.index += 1
    summary_df.columns = ['Ticker', '銘柄名', 'トレード数', 'ストップ高数', '勝率(%)', '平均損益(%)', '最大含み益(%)', '最大含み損(%)']
    print(summary_df.to_string())

    # --------------------------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------------------------
    output_filename = "stophigh_backtest_result.csv"
    df_all.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"\n{'='*80}")
    print(f"★ 全 {len(df_all)} 件の詳細ログを '{output_filename}' にCSV保存しました。")
    print(f"  ※ is_stop_high 列で通常/ストップ高ケースを区別できます。")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_multi_ticker_backtest()
