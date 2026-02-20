import asyncio
import requests
import pandas as pd
import numpy as np
import ta
import joblib
import os
import sqlite3
from xgboost import XGBClassifier
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, JobQueue

TOKEN = "8237062025:AAFv6__wBeZDmur8kcEHVjKIQblbwmK-lWY"
TWELVE_KEY = "413f1870be274f7fbfff5ab5d720c5a5"
DB_NAME = "xauusd_ai.db"
MODEL_FILE = "xauusd_model.pkl"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, direction TEXT, entry REAL, result INTEGER)")
    conn.commit()
    conn.close()

def fetch_data(interval="15min", outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol":"XAU/USD","interval":interval,"outputsize":outputsize,"apikey":TWELVE_KEY}
    r = requests.get(url, params=params).json()
    if "values" not in r:
        raise Exception("Twelve Data API Error: "+str(r))
    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    numeric_cols = ["open","high","low","close","volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df.iloc[::-1].reset_index(drop=True)
    return df

def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"],14).rsi()
    df["ema50"] = ta.trend.EMAIndicator(df["close"],50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"],200).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"],df["low"],df["close"],14).average_true_range()
    df["bop"] = (df["close"]-df["open"])/(df["high"]-df["low"]+1e-6)
    df["psar"] = ta.trend.PSARIndicator(df["high"],df["low"],df["close"],step=0.02,max_step=0.2).psar()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["bull_engulf"] = ((df["close"]>df["open"]) & (df["close"].shift(1)<df["open"].shift(1)) & (df["close"]>df["open"].shift(1))).astype(int)
    df["bear_engulf"] = ((df["close"]<df["open"]) & (df["close"].shift(1)>df["open"].shift(1)) & (df["close"]<df["open"].shift(1))).astype(int)
    df.dropna(inplace=True)
    return df

def detect_structure(df,swing=3):
    if len(df)<swing+1: return 0
    highs = df["high"].iloc[-swing:]
    lows = df["low"].iloc[-swing:]
    if df["close"].iloc[-1] > max(highs[:-1]) and df["close"].iloc[-2] < max(highs[:-1]): return 1
    if df["close"].iloc[-1] < min(lows[:-1]) and df["close"].iloc[-2] > min(lows[:-1]): return -1
    return 0

def detect_liquidity_sweep(df,threshold=0.5):
    if len(df)<3: return 0
    prev_high = df["high"].iloc[-2]
    prev_low = df["low"].iloc[-2]
    ch = df["high"].iloc[-1]
    cl = df["low"].iloc[-1]
    cc = df["close"].iloc[-1]
    body = abs(df["close"].iloc[-1]-df["open"].iloc[-1])
    if ch>prev_high and cc<prev_high and body>threshold*df["atr"].iloc[-2]: return 1
    if cl<prev_low and cc>prev_low and body>threshold*df["atr"].iloc[-2]: return -1
    return 0

def supply_demand(df):
    if len(df)<3: return 0
    body = abs(df["close"].iloc[-2]-df["open"].iloc[-2])
    if body>df["atr"].iloc[-2]:
        if df["close"].iloc[-2]>df["open"].iloc[-2]: return 1
        return -1
    return 0

def train_model(df):
    df["target"] = (df["close"].shift(-3)>df["close"]).astype(int)
    df.dropna(inplace=True)
    features = ["rsi","bop","ema50","ema200","atr","psar","macd","macd_signal","bull_engulf","bear_engulf"]
    X = df[features]
    y = df["target"]
    if len(np.unique(y))<2: return
    model = XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,use_label_encoder=False,eval_metric="logloss")
    model.fit(X,y)
    joblib.dump(model,MODEL_FILE)

def load_model(df):
    if not os.path.exists(MODEL_FILE): train_model(df)
    if os.path.exists(MODEL_FILE): return joblib.load(MODEL_FILE)
    return None

def predict(df,model):
    if model is None: return 0.5
    features = ["rsi","bop","ema50","ema200","atr","psar","macd","macd_signal","bull_engulf","bear_engulf"]
    latest = df[features].iloc[-1:]
    prob = model.predict_proba(latest)[0][1]
    return prob

def generate_signal(df1m,df15m,df1h,model):
    bias = detect_structure(df1h)
    structure15 = detect_structure(df15m)
    sweep = detect_liquidity_sweep(df15m)
    zone = supply_demand(df15m)
    psar_candle = df1m["psar"].iloc[-1]
    macd_c = df1m["macd"].iloc[-1]
    macd_signal_c = df1m["macd_signal"].iloc[-1]
    entry_candle = df1m.iloc[-1]
    confidence = predict(df1m,model)
    direction = "BUY"
    if bias<0: direction="SELL"
    if structure15<0: direction="SELL"
    if structure15>0: direction="BUY"
    if sweep==-1: direction="SELL"
    if sweep==1: direction="BUY"
    if zone==-1: direction="SELL"
    if zone==1: direction="BUY"
    if direction=="BUY" and entry_candle["close"]<psar_candle: direction="SELL"
    if direction=="SELL" and entry_candle["close"]>psar_candle: direction="BUY"
    if direction=="BUY" and macd_c<macd_signal_c: direction="SELL"
    if direction=="SELL" and macd_c>macd_signal_c: direction="BUY"
    entry = entry_candle["close"]
    tp1 = entry+1.0 if direction=="BUY" else entry-1.0
    tp2 = entry+1.6 if direction=="BUY" else entry-1.6
    sl = entry-0.8 if direction=="BUY" else entry+0.8
    return direction,entry,tp1,tp2,sl,round(confidence*100,2)

def backtest(df,model):
    wins = 0
    total = 0
    for i in range(210,len(df)-5):
        sub = df.iloc[:i].copy()
        direction,entry,tp1,tp2,sl,_ = generate_signal(sub,sub,sub,model)
        future = df["close"].iloc[i+3]
        total+=1
        if direction=="BUY" and future>entry: wins+=1
        if direction=="SELL" and future<entry: wins+=1
    if total==0: return 0
    return round((wins/total)*100,2)

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("Get Signal",callback_data="signal")],
                [InlineKeyboardButton("Backtest",callback_data="backtest")]]
    await update.message.reply_text("XAUUSD Multi-TF AI with SAR & MACD Ready",reply_markup=InlineKeyboardMarkup(keyboard))

async def button(update:Update,context:ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        df1m = add_indicators(fetch_data("1min",500))
        df15m = add_indicators(fetch_data("15min",500))
        df1h = add_indicators(fetch_data("1h",500))
        model = load_model(df1m)
        if query.data=="signal":
            direction,entry,tp1,tp2,sl,confidence = generate_signal(df1m,df15m,df1h,model)
            text = f"XAUUSD {direction}\nEntry:{entry}\nTP1:{tp1}\nTP2:{tp2}\nSL:{sl}\nConfidence:{confidence}%"
            await query.edit_message_text(text)
        if query.data=="backtest":
            df = df15m
            result = backtest(df,model)
            await query.edit_message_text(f"Win Rate: {result}%")
    except Exception as e:
        await query.edit_message_text(str(e))

async def subscribe(update:Update,context:ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if "subscribers" not in context.application.bot_data: context.application.bot_data["subscribers"]=[]
    if chat_id not in context.application.bot_data["subscribers"]: context.application.bot_data["subscribers"].append(chat_id)
    await update.message.reply_text("Subscribed to auto signals")

async def auto_signals(context:ContextTypes.DEFAULT_TYPE):
    try:
        df1m = add_indicators(fetch_data("1min",500))
        df15m = add_indicators(fetch_data("15min",500))
        df1h = add_indicators(fetch_data("1h",500))
        model = load_model(df1m)
        direction,entry,tp1,tp2,sl,confidence = generate_signal(df1m,df15m,df1h,model)
        text = f"Auto Signal\nXAUUSD {direction}\nEntry:{entry}\nTP1:{tp1}\nTP2:{tp2}\nSL:{sl}\nConfidence:{confidence}%"
        subscribers = context.application.bot_data.get("subscribers",[])
        for chat_id in subscribers: await context.bot.send_message(chat_id,text)
        train_model(df1m)
    except: pass

init_db()
app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start",start))
app.add_handler(CommandHandler("subscribe",subscribe))
app.add_handler(CallbackQueryHandler(button))
app.run_polling()