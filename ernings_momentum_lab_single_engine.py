# 好決算後の翌日寄り付き買い（決算モメンタム）複数銘柄一括バックテストスクリプト (Groq API ログ強化版)

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
# ------------------------------------------------------------------------------
# 設定パラメータ (.env から自動ロード。未設定時はデフォルト値)
# ------------------------------------------------------------------------------
TICKER_LIST_FILE = os.getenv("TICKER_LIST_FILE", "ticker_list.txt")  # 読み込む銘柄リストファイル
BACKTEST_YEARS = int(os.getenv("BACKTEST_YEARS", "5"))             # 過去何年分を対象にするか
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "80"))        # エントリー対象とするAI好決算スコアの閾値
HOLD_DAYS = int(os.getenv("HOLD_DAYS", "5"))                     # 保有日数

groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
    print("【エラー】.env ファイルに GROQ_API_KEY が設定されていません。")
    exit(1)

client = Groq(api_key=groq_key)

# ------------------------------------------------------------------------------
# 2. テキスト生成（LLM）専用モデルの自動検出
# ------------------------------------------------------------------------------
def get_working_groq_llm_model() -> str:
    """Groqからチャット生成(LLM)に対応した有効モデルを自動取得"""
    try:
        models_list = client.models.list()
        available_models = [m.id for m in models_list.data]

        # 音声・ガード専用モデルを除外
        llm_models = [
            m for m in available_models 
            if not any(invalid in m.lower() for invalid in ["whisper", "guard", "embedding", "vision"])
        ]

        preferred_keywords = ["llama-3.3-70b", "llama-3.1-8b", "llama3-70b", "llama3-8b", "mixtral", "gpt-oss"]
        for pref in preferred_keywords:
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
# 3. AI（Groq API）決算評価関数
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
# 4. 単一銘柄のバックテスト処理 (MFE / MAE 計算追加)
# ------------------------------------------------------------------------------
def backtest_single_ticker(ticker: str, years: int, score_threshold: int, hold_days: int) -> list:
    stock = yf.Ticker(ticker)
    
    # 会社名の取得（ログ用）
    try:
        company_name = stock.info.get("shortName") or stock.info.get("longName") or ticker
    except Exception:
        company_name = ticker

    # 株価取得
    df_price = stock.history(period=f"{years}y")
    if df_price.empty:
        print(f"  └─ ⚠️ 株価データが取得できませんでした。スキップします。")
        return []

    df_price = df_price.reset_index()
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.tz_localize(None)
    df_price = df_price.sort_values('Date').reset_index(drop=True)

    # 決算日取得
    try:
        earnings_dates = stock.earnings_dates
        if earnings_dates is None or earnings_dates.empty:
            print(f"  └─ ⚠️ 決算日データが存在しません。スキップします。")
            return []
    except Exception:
        print(f"  └─ ⚠️ 決算日データの取得に失敗しました。スキップします。")
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
        exit_idx = entry_idx + (hold_days - 1)

        # 保有期間中の株価スライス (エントリー日〜イグジット日)
        hold_period_df = df_price.loc[entry_idx:exit_idx]

        entry_date = df_price.loc[entry_idx, 'Date'].strftime('%Y-%m-%d')
        exit_date = df_price.loc[exit_idx, 'Date'].strftime('%Y-%m-%d')
        
        entry_price = df_price.loc[entry_idx, 'Open']
        exit_price = df_price.loc[exit_idx, 'Close']

        # 最終損益率
        return_pct = ((exit_price - entry_price) / entry_price) * 100

        # 保有期間中の最大含み益率 (MFE) と 最大含み損率 (MAE) の計算
        max_high = hold_period_df['High'].max()
        min_low = hold_period_df['Low'].min()
        mfe_pct = ((max_high - entry_price) / entry_price) * 100
        mae_pct = ((min_low - entry_price) / entry_price) * 100

        trade_record = {
            'ticker': ticker,
            'name': company_name,
            'earnings_date': e_date_str,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'score': score,
            'entry_price': round(entry_price, 1),
            'exit_price': round(exit_price, 1),
            'return_pct': round(return_pct, 2),
            'mfe_pct': round(mfe_pct, 2),
            'mae_pct': round(mae_pct, 2)
        }
        ticker_trades.append(trade_record)

        # 個別シミュレーション結果ログ出力
        print(f"[{ticker}:{company_name}]  [決算日: {e_date_str}] AIスコア: {score}点 -> 翌日Open買 ({entry_price:.1f}円) ➔ {hold_days}日後Close売 ({exit_price:.1f}円) | 損益: {return_pct:+.2f}% (最大含み益: +{mfe_pct:.2f}% / 最大含み損: {mae_pct:.2f}%)")

        time.sleep(1)

    return ticker_trades

# ------------------------------------------------------------------------------
# 5. 全銘柄ループ & ポートフォリオランキング集計
# ------------------------------------------------------------------------------
def run_multi_ticker_backtest():
    if not os.path.exists(TICKER_LIST_FILE):
        print(f"【エラー】リストファイル '{TICKER_LIST_FILE}' が見つかりません。")
        return

    with open(TICKER_LIST_FILE, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"\n================================================================================")
    print(f"  複数銘柄一括バックテスト開始 (全 {len(tickers)} 銘柄 / 過去 {BACKTEST_YEARS} 年間 / 保有 {HOLD_DAYS} 日)")
    print(f"================================================================================\n")

    all_trades = []

    for i, ticker in enumerate(tickers, 1):
        print(f"--------------------------------------------------------------------------------")
        print(f" [{i}/{len(tickers)}] {ticker} の検証処理中...")
        print(f"--------------------------------------------------------------------------------")
        
        trades = backtest_single_ticker(ticker, BACKTEST_YEARS, SCORE_THRESHOLD, HOLD_DAYS)
        if trades:
            all_trades.extend(trades)
            win_count = sum(1 for t in trades if t['return_pct'] > 0)
            win_rate = (win_count / len(trades)) * 100
            avg_ret = sum(t['return_pct'] for t in trades) / len(trades)
            print(f"  └─ 完了: {len(trades)}戦{win_count}勝{len(trades)-win_count}敗 (勝率 {win_rate:.1f}% / 平均損益 {avg_ret:+.2f}%)\n")
        else:
            print(f"  └─ 条件を満たすトレードシグナルはありませんでした。\n")

    # --------------------------------------------------------------------------
    # 最終集計レポート
    # --------------------------------------------------------------------------
    print("\n================================================================================")
    print("   【全銘柄 統合ポートフォリオ パフォーマンスランキング】")
    print("================================================================================")

    if not all_trades:
        print("全体で条件を満たすトレードは1件も発生しませんでした。")
        return

    df_all = pd.DataFrame(all_trades)

    win_trades = df_all[df_all['return_pct'] > 0]
    overall_win_rate = (len(win_trades) / len(df_all)) * 100
    overall_avg_return = df_all['return_pct'].mean()
    overall_max_return = df_all['return_pct'].max()
    overall_min_return = df_all['return_pct'].min()

    print(f"総検証銘柄数 : {len(tickers)} 銘柄")
    print(f"総トレード数 : {len(df_all)} 回")
    print(f"全体勝率     : {overall_win_rate:.1f} % ({len(win_trades)}勝 {len(df_all)-len(win_trades)}敗)")
    print(f"全体平均損益 : {overall_avg_return:+.2f} %")
    print(f"最大利益率   : {overall_max_return:+.2f} %")
    print(f"最大損失率   : {overall_min_return:+.2f} %")

    # 銘柄別の集計内訳（平均損益の降順ランキング）
    print("\n--- 銘柄別パフォーマンスランキング（平均損益順） ---")
    summary_df = df_all.groupby(['ticker', 'name']).agg(
        トレード数=('return_pct', 'count'),
        勝率_pct=('return_pct', lambda x: round((x > 0).mean() * 100, 1)),
        平均損益_pct=('return_pct', lambda x: round(x.mean(), 2)),
        最大利幅_pct=('mfe_pct', lambda x: round(x.max(), 2)),
        最大掘幅_pct=('mae_pct', lambda x: round(x.min(), 2))
    ).reset_index()

    # 平均損益率の降順に並び替え
    summary_df = summary_df.sort_values(by='平均損益_pct', ascending=False).reset_index(drop=True)
    summary_df.index += 1  # 順位表示用（1位〜）

    # 列名の調整とコンソール表示
    summary_df.columns = ['Ticker', '銘柄名', 'トレード数', '勝率(%)', '平均損益(%)', '最大含み益(%)', '最大含み損(%)']
    print(summary_df.to_string())

    # CSV出力保存
    output_filename = "multi_backtest_result.csv"
    df_all.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"\n================================================================ shadows")
    print(f"★ 全 {len(df_all)} 件の詳細ログを '{output_filename}' にCSV保存しました。")
    print(f"========================================================================")

if __name__ == "__main__":
    run_multi_ticker_backtest()