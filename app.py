import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Streamlit 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Gemini 30년차 트레이더 모의투자", layout="wide")
st.title("⚡ Gemini 30년차 전문 트레이더 모의투자 시스템")

# API 키 가져오기 (Streamlit Secrets 또는 환경 변수)
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = None
    st.warning("⚠️ Secrets에 GEMINI_API_KEY가 설정되지 않았습니다.")

TRADER_SYSTEM_INSTRUCTION = """
너는 30년차 전문 트레이더야. 주식을 볼 때 일봉, 거래량, 매물대, 볼린저 밴드, 이동 평균선 전체, AD라인, 미국의 통화정책, 경제 뉴스를 보면서 투자를 해.
너의 목표 수익률은 1년에 200%야, 공격적인 투자자이지.
단호하고 날카로운 트레이더의 톤으로 분석 결과를 전달해라.
"""

# ---------------------------------------------------------
# 데이터 수집 및 계산 함수
# ---------------------------------------------------------
@st.cache_data(ttl=3600)  # 1시간 캐싱
def fetch_stock_data(symbol_name):
    symbol_map = {"이더리움": "ETH-USD", "SOXL": "SOXL", "KORU": "KORU", "URAA": "URA"}
    ticker = symbol_map.get(symbol_name, symbol_name)
    
    df = yf.download(ticker, period="120d", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna()

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['Vol_Avg_90'] = df['Volume'].rolling(window=90).mean()

    df_90 = df.tail(90).copy()
    counts, bin_edges = np.histogram(df_90['Close'], bins=10, weights=df_90['Volume'])
    vp_top = bin_edges[np.argmax(counts) + 1]

    return df_90, vp_top

@st.cache_data(ttl=3600)
def fetch_market_indices():
    tickers = {"나스닥": "^IXIC", "S&P500": "^GSPC", "코스피": "^KS11", "이더리움": "ETH-USD", "원/달러 환율": "KRW=X"}
    data = {}
    for name, sym in tickers.items():
        t_data = yf.Ticker(sym).history(period="2d")
        if len(t_data) >= 2:
            close = t_data['Close'].iloc[-1]
            prev = t_data['Close'].iloc[-2]
            data[name] = {"price": close, "change": ((close - prev) / prev) * 100}
        else:
            data[name] = {"price": 0.0, "change": 0.0}
    return data

# ---------------------------------------------------------
# 가상 매매 엔진 Class
# ---------------------------------------------------------
class SimulatedTrader:
    def __init__(self, initial_balance=20000000):
        self.cash = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.trade_logs = []

    def execute_strategy(self, symbol, df, vp_top):
        latest = df.iloc[-1]
        price = latest['Close']
        ma5, ma20 = latest['MA5'], latest['MA20']
        vol, vol_avg_90 = latest['Volume'], latest['Vol_Avg_90']
        bb_lower = latest['BB_Lower']
        
        pos = self.positions.get(symbol, {'qty': 0, 'avg_price': 0.0, 'bb_stage': 0})
        
        # 볼린저 밴드 하한선 물타기 로직
        if latest['Low'] <= bb_lower:
            stage = pos['bb_stage']
            if stage == 0:
                self._buy(symbol, price, self.cash * 0.30, "BB 1차 하한선 (잔고 30% / 목표익절 +3%)", 1)
                return
            elif stage == 1:
                self._buy(symbol, price, self.cash * 0.50, "BB 2차 하한선 (잔고 50% / 목표익절 +1%)", 2)
                return
            elif stage >= 2:
                self._buy(symbol, price, self.cash * 0.50, "BB 3차 하한선 (잔고 50% / 목표익절 0%)", 3)
                return

        if pos['qty'] > 0 and pos['bb_stage'] > 0:
            roi = ((price - pos['avg_price']) / pos['avg_price']) * 100
            target = {1: 3.0, 2: 1.0, 3: 0.0}.get(pos['bb_stage'], 0.0)
            if roi >= target:
                self._sell(symbol, price, f"BB {pos['bb_stage']}단계 목표 수익률({target}%) 달성 매도")
                return

        # 일반 조건
        cond1 = (price > vp_top) and (ma5 > ma20)
        cond2 = (vol > vol_avg_90 * 1.5) and (ma5 > ma20)
        
        if pos['qty'] == 0:
            if cond1:
                self._buy(symbol, price, self.cash * 0.25, "매물대 돌파 및 MA5 > MA20", 0)
            elif cond2:
                self._buy(symbol, price, self.cash * 0.25, "거래량 1.5배 돌파 및 MA5 > MA20", 0)
        elif pos['qty'] > 0 and pos['bb_stage'] == 0:
            if ma20 > ma5:
                self._sell(symbol, price, "MA20 > MA5 데드크로스 매도")

    def _buy(self, symbol, price, amount, reason, bb_stage):
        if amount < 10000: return
        qty = amount / price
        pos = self.positions.get(symbol, {'qty': 0, 'avg_price': 0.0, 'bb_stage': 0})
        total_qty = pos['qty'] + qty
        total_cost = (pos['qty'] * pos['avg_price']) + amount
        self.cash -= amount
        self.positions[symbol] = {'qty': total_qty, 'avg_price': total_cost / total_qty, 'bb_stage': bb_stage}
        self.trade_logs.append({"타입": "매수", "종목": symbol, "가격": round(price, 2), "수량": round(qty, 4), "수익률(%)": 0.0, "매매 이유": reason})

    def _sell(self, symbol, price, reason):
        pos = self.positions.get(symbol)
        if not pos or pos['qty'] == 0: return
        sell_val = pos['qty'] * price
        cost_val = pos['qty'] * pos['avg_price']
        roi = ((sell_val - cost_val) / cost_val) * 100
        self.cash += sell_val
        del self.positions[symbol]
        self.trade_logs.append({"타입": "매도", "종목": symbol, "가격": round(price, 2), "수량": round(pos['qty'], 4), "수익률(%)": round(roi, 2), "매매 이유": reason})

# 세션 상태 관리
if "trader" not in st.session_state:
    st.session_state.trader = SimulatedTrader()

trader = st.session_state.trader
target_symbols = ["이더리움", "SOXL", "KORU", "URAA"]
stock_datas = {}
total_eval = 0.0

for sym in target_symbols:
    df, vp_top = fetch_stock_data(sym)
    stock_datas[sym] = (df, vp_top)
    trader.execute_strategy(sym, df, vp_top)
    if sym in trader.positions:
        total_eval += trader.positions[sym]['qty'] * df['Close'].iloc[-1]

# 1. 계좌 현황 UI
total_assets = trader.cash + total_eval
total_profit = total_assets - trader.initial_balance
total_roi = (total_profit / trader.initial_balance) * 100

st.subheader("💼 계좌 현황")
c1, c2, c3, c4 = st.columns(4)
c1.metric("계좌 잔고", f"{int(trader.cash):,} 원")
c2.metric("평가 금액", f"{int(total_eval):,} 원")
c3.metric("수익금", f"{int(total_profit):,} 원", delta=f"{int(total_profit):,} 원")
c4.metric("수익률", f"{total_roi:.2f} %", delta=f"{total_roi:.2f}%")

st.markdown("---")

# 2. 시장 지수 및 Gemini 시황 분석 UI
st.subheader("🌐 대표 시장 지수")
indices = fetch_market_indices()
idx_cols = st.columns(len(indices))
for idx, (name, val) in enumerate(indices.items()):
    idx_cols[idx].metric(label=name, value=f"{val['price']:,.2f}", delta=f"{val['change']:.2f}%")

st.markdown("### 📊 장 마감 제미나이 트레이더 시황 분석")
if st.button("제미나이 AI 지수 분석 실행"):
    if client:
        with st.spinner("30년차 트레이더 분석 중..."):
            prompt = f"지수 현황: {indices}. 오늘 장 마감 후 시장 요인을 3줄로 날카롭게 요약해라."
            res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=types.GenerateContentConfig(system_instruction=TRADER_SYSTEM_INSTRUCTION))
            st.info(res.text)
    else:
        st.error("Gemini API 키가 연결되지 않았습니다.")

st.markdown("---")

# 3. 차트 시각화 UI
st.subheader("📈 설정 종목 일봉 분석 (최근 90일)")
selected_stock = st.selectbox("종목 선택", target_symbols)
df_selected, vp_top_selected = stock_datas[selected_stock]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
fig.add_trace(go.Candlestick(x=df_selected.index, open=df_selected['Open'], high=df_selected['High'], low=df_selected['Low'], close=df_selected['Close'], name="주가"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['MA5'], line=dict(color='orange', width=1), name="MA5"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['MA20'], line=dict(color='blue', width=1.5), name="MA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['BB_Upper'], line=dict(color='gray', dash='dash'), name="BB 상한"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['BB_Lower'], line=dict(color='gray', dash='dash'), name="BB 하한"), row=1, col=1)
fig.add_hline(y=vp_top_selected, line_color="red", line_dash="dot", annotation_text=f"매물대 상단 ({vp_top_selected:.2f})", row=1, col=1)

fig.add_trace(go.Bar(x=df_selected.index, y=df_selected['Volume'], name="거래량", marker_color='purple'), row=2, col=1)
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['Vol_Avg_90'], line=dict(color='red', width=1), name="90일 평균 거래량"), row=2, col=1)

fig.update_layout(xaxis_rangeslider_visible=False, height=550)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 4. 매매일지 UI
st.subheader("📑 주식 매매 일지")
if trader.trade_logs:
    st.dataframe(pd.DataFrame(trader.trade_logs), use_container_width=True)
else:
    st.write("현재까지 실행된 매매 내역이 없습니다.")
    