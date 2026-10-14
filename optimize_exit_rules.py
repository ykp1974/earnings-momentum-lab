# イグジットルール（利確/損切りパラメータ）グリッドサーチ最適化スクリプト

import os
import json
import time
import pandas as pd
import yfinance as yf
from groq import Groq
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 0. 初期設定 & 環境変数ロード
# ------------------------------------------------------------------------------
load_dotenv()
# ------------------------------------------------------------------------------
# 設定パラメータ (.env から自動ロード)
# ------------------------------------------------------------------------------
TICKER_LIST_FILE = os.getenv("TICKER_LIST_FILE", "ticker_list.txt") # 検証対象銘柄リスト
BACKTEST_YEARS = int(os.getenv("BACKTEST_YEARS", "5")) # 対象期間（年）
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "80")) # AI好決算スコアの閾値
MAX_HOLD_DAYS = int(os.getenv("HOLD_DAYS", "5"))  # 最大保有日数(HOLD_DAYS の値を適用)

groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
    print("【エラー】.env ファイルに GROQ_API_KEY が設定されていません。")
    exit(1)

client = Groq(api_key=groq_key)

# ★ 探索するイグジットパラメータのパターン
STOP_LOSS_GRID = [-2.0, -3.0, -4.0, -5.0, None]       # 損切りライン (Noneは損切りなし)
TAKE_PROFIT_GRID = [3.0, 5.0, 8.0, 10.0, 12.0, None]   # 利確ライン (Noneは利確なし)

# ------------------------------------------------------------------------------
# 2. テキスト生成（LLM）専用モデルの自動検出
# ------------------------------------------------------------------------------
def get_working_groq_llm_model() -> str:
    try:
        models_list = client.models.list()
        available_models = [m.id for m in models_list.data]
        llm_models = [
            m for m in available_models 
            if not any(invalid in m.lower() for invalid in ["whisper", "guard", "embedding", "vision"])
        ]
        preferred_keywords = ["llama-3.3-70b", "llama-3.1-8b", "llama3-70b", "llama3-8b", "mixtral", "gpt-oss"]
        for pref in preferred_keywords:
            for model_id in llm_models:
                if pref in model_id:
                    return model_id
        if llm_models:
            return llm_models[0]
    except Exception:
        pass
    return "llama-3.1-8b-instant"

ACTIVE_MODEL = get_working_groq_llm_model()

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
    except Exception:
        return 0

# ------------------------------------------------------------------------------
# 3. エントリーイベント（シグナル）の全件収集
# ------------------------------------------------------------------------------
def collect_all_entry_signals():
    """まず全銘柄の「決算エントリーシグナル」と「その後の日足価格履歴」を一括収集する"""
    if not os.path.exists(TICKER_LIST_FILE):
        print(f"【エラー】リストファイル '{TICKER_LIST_FILE}' が見つかりません。")
        return []

    with open(TICKER_LIST_FILE, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"=== [Step 1] 対象 {len(tickers)} 銘柄からAIエントリーシグナルを抽出中 (Groq: {ACTIVE_MODEL}) ===")
    
    signals = []

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df_price = stock.history(period=f"{BACKTEST_YEARS}y")
        if df_price.empty:
            continue

        df_price = df_price.reset_index()
        df_price['Date'] = pd.to_datetime(df_price['Date']).dt.tz_localize(None)
        df_price = df_price.sort_values('Date').reset_index(drop=True)

        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is None or earnings_dates.empty:
                continue
        except Exception:
            continue

        earnings_dates = earnings_dates.reset_index()
        earnings_dates['Earnings Date'] = pd.to_datetime(earnings_dates['Earnings Date']).dt.tz_localize(None)
        past_earnings = earnings_dates[earnings_dates['Earnings Date'] <= df_price['Date'].max()].copy()

        for idx, row in past_earnings.iterrows():
            e_date = row['Earnings Date'].normalize()
            e_date_str = e_date.strftime('%Y-%m-%d')
            dummy_earnings_text = f"第{idx+1}四半期決算短信：営業利益 前年同期比 +25%増益。通期予想を上方修正。"

            score = evaluate_earnings_with_groq(ticker, e_date_str, dummy_earnings_text)
            if score < SCORE_THRESHOLD:
                continue

            future_prices = df_price[df_price['Date'] > e_date]
            if len(future_prices) < MAX_HOLD_DAYS:
                continue

            entry_idx = future_prices.index[0]
            exit_limit_idx = entry_idx + (MAX_HOLD_DAYS - 1)
            hold_df = df_price.loc[entry_idx:exit_limit_idx].copy().reset_index(drop=True)

            signals.append({
                'ticker': ticker,
                'earnings_date': e_date_str,
                'score': score,
                'entry_price': hold_df.loc[0, 'Open'],
                'price_history': hold_df  # 保有期間5日間の日足データ (Open, High, Low, Close)
            })
            time.sleep(1)

    print(f" -> 合計 {len(signals)} 件の有効エントリーイベントを抽出しました。\n")
    return signals

# ------------------------------------------------------------------------------
# 4. イグジットルールのグリッドサーチシミュレーション
# ------------------------------------------------------------------------------
def simulate_exit_strategy(signals: list, stop_loss_pct: float, take_profit_pct: float) -> dict:
    """特定の（損切り%, 利確%）ルール下での損益パフォーマンスを追跡計算"""
    returns = []

    for sig in signals:
        entry_price = sig['entry_price']
        price_history = sig['price_history']

        exit_price = None
        exit_reason = "TIME_EXIT"  # デフォルトは日形満了

        # 5日間の日足を順番にチェック
        for day_idx, row in price_history.iterrows():
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']

            # 1. 損切り判定 (Lowで判定)
            if stop_loss_pct is not None:
                stop_price = entry_price * (1 + stop_loss_pct / 100.0)
                if low_price <= stop_price:
                    exit_price = stop_price  # 損切り成行約定
                    exit_reason = "STOP_LOSS"
                    break

            # 2. 利確判定 (Highで判定)
            if take_profit_pct is not None:
                target_price = entry_price * (1 + take_profit_pct / 100.0)
                if high_price >= target_price:
                    exit_price = target_price  # 利確指値約定
                    exit_reason = "TAKE_PROFIT"
                    break

        # 利確・損切りに掛からず5日経過した場合は、最終日のCloseで手仕舞い
        if exit_price is None:
            exit_price = price_history.iloc[-1]['Close']

        ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
        returns.append(ret_pct)

    df_ret = pd.Series(returns)
    win_trades = df_ret[df_ret > 0]
    
    win_rate = (len(win_trades) / len(df_ret)) * 100.0 if len(df_ret) > 0 else 0
    avg_return = df_ret.mean() if len(df_ret) > 0 else 0
    total_return = df_ret.sum() if len(df_ret) > 0 else 0
    max_drawdown = df_ret.min() if len(df_ret) > 0 else 0

    return {
        'stop_loss': f"{stop_loss_pct}%" if stop_loss_pct else "なし",
        'take_profit': f"+{take_profit_pct}%" if take_profit_pct else "なし",
        'total_trades': len(df_ret),
        'win_rate_pct': round(win_rate, 1),
        'avg_return_pct': round(avg_return, 2),
        'total_return_pct': round(total_return, 2),
        'max_loss_pct': round(max_drawdown, 2)
    }

# ------------------------------------------------------------------------------
# 5. メイン実行エンジン
# ------------------------------------------------------------------------------
def run_optimization():
    signals = collect_all_entry_signals()
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
    df_sorted = df_grid.sort_values(by="avg_return_pct", ascending=False).reset_index(drop=True)
    df_sorted.index += 1

    print("\n================================================================================")
    print("   【イグジットルール 最適パラメータ パフォーマンスランキング】")
    print("================================================================================")
    print(f"検証イベント数 : 全 {len(signals)} トレード")
    print(df_sorted.to_string())

    # 最優良設定の抽出・表示
    best_config = df_sorted.iloc[0]
    print("\n--------------------------------------------------------------------------------")
    print("★【最適解 (Best Strategy)】")
    print(f"  ・損切りライン (Stop Loss)   : {best_config['stop_loss']}")
    print(f"  ・利確ライン   (Take Profit) : {best_config['take_profit']}")
    print(f"  ・勝率                       : {best_config['win_rate_pct']} %")
    print(f"  ・平均損益率                 : {best_config['avg_return_pct']:+} %")
    print(f"  ・最大損失率 (1トレード)     : {best_config['max_loss_pct']} %")
    print("--------------------------------------------------------------------------------\n")

    # 結果をCSV出力
    df_sorted.to_csv("exit_rule_optimization_result.csv", index=False, encoding="utf-8-sig")
    print("★ 詳細結果を 'exit_rule_optimization_result.csv' に出力保存しました。")

if __name__ == "__main__":
    run_optimization()