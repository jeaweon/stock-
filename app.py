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
# Streamlit 기본 설정 & 1시간(3600초) 자동 새로고침 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Gemini 30년차 트레이더 모의투자", layout="wide")
st.title("⚡ Gemini 30년차 전문 트레이더 모의투자 시스템")

st.caption("🔄 데이터 갱신 주기: 1시간 단위 자동 업데이트")

# API 키 가져오기
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
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol_name):
    symbol_map = {
        "이더리움": "ETH-USD",
        "SOXL": "SOXL",
        "KORU": "KORU",
        "URAA": "URA",
        "팔란티어": "PLTR",
        "로켓 랩": "RKLB",
        "슈퍼 마이크로 컴퓨터": "SMCI",
        "알파벳": "GOOGL",
        "애플": "AAPL",
        "아이온큐": "IONQ",
        "퀄컴": "QCOM"
    }
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
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20']
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

       # (기존) 볼린저 밴드 하한선 물타기 로직 및 익절 로직 유지...
        
        # ---------------------------------------------------------
        # [신규 최우선 조건 1] VCP (변동성 수축 후 폭발)
        # ---------------------------------------------------------
        # 전일 기준 볼린저 밴드 폭이 10% 미만(0.10)으로 극도로 수축되었고, 
        # 당일 거래량이 90일 평균의 3배 이상 터지며 볼린저 밴드 상단 돌파
        prev_bb_width = df['BB_Width'].iloc[-2]
        cond_vcp = (prev_bb_width < 0.10) and (vol >= vol_avg_90 * 3.0) and (price > latest['BB_Upper'])

        # ---------------------------------------------------------
        # [신규 최우선 조건 2] 핵심 매물대 풀백 (Pullback & Bounce)
        # ---------------------------------------------------------
        # 당일 저가가 매물대 상단(vp_top)의 2% 이내로 근접하여 지지 테스트를 마치고,
        # 종가가 시가보다 높은 양봉(Close > Open)으로 마감하며 5일선이 20일선 위에 있을 때
        is_yangbong = latest['Close'] > latest['Open']
        touched_support = latest['Low'] <= (vp_top * 1.02)
        cond_pullback = is_yangbong and touched_support and (price > vp_top) and (ma5 > ma20)

        # ---------------------------------------------------------
        # (기존) 일반 돌파 매수 조건
        # ---------------------------------------------------------
        cond1 = (price > vp_top) and (ma5 > ma20)
        cond2 = (vol > vol_avg_90 * 1.5) and (ma5 > ma20)
        
        if pos['qty'] == 0:
            if cond_vcp:
                self._buy(symbol, price, self.cash * 0.40, "🔥 VCP 돌파: BB 극도 수축 후 3배 거래량 폭발", 0)
            elif cond_pullback:
                self._buy(symbol, price, self.cash * 0.35, "🎯 매물대 풀백: 핵심 저항 지지 후 첫 양봉 반등", 0)
            elif cond1:
                self._buy(symbol, price, self.cash * 0.20, "매물대 돌파 및 MA5 > MA20", 0)
            elif cond2:
                self._buy(symbol, price, self.cash * 0.20, "거래량 1.5배 돌파 및 MA5 > MA20", 0)
                
        elif pos['qty'] > 0 and pos['bb_stage'] == 0:
            if ma20 > ma5:
                self._sell(symbol, price, "MA20 > MA5 데드크로스 매도")

    # (SimulatedTrader 클래스 내부의 기존 _buy, _sell 함수 덮어쓰기)
    def _buy(self, symbol, price, amount, reason, bb_stage):
        if amount < 10000: return
        qty = amount / price
        pos = self.positions.get(symbol, {'qty': 0, 'avg_price': 0.0, 'bb_stage': 0})
        
        # 최초 매수 시점의 전략(reason)을 포지션에 기록하여 승률 추적에 사용
        entry_strategy = reason if pos['qty'] == 0 else pos.get('entry_strategy', reason)
        
        total_qty = pos['qty'] + qty
        total_cost = (pos['qty'] * pos['avg_price']) + amount
        self.cash -= amount
        
        self.positions[symbol] = {
            'qty': total_qty, 
            'avg_price': total_cost / total_qty, 
            'bb_stage': bb_stage, 
            'entry_strategy': entry_strategy
        }
        
        self.trade_logs.append({
            "타입": "매수", "종목": symbol, "가격": round(price, 2), 
            "수량": round(qty, 4), "수익률(%)": 0.0, 
            "진입 전략": entry_strategy, "상세 사유": reason
        })

    def _sell(self, symbol, price, reason):
        pos = self.positions.get(symbol)
        if not pos or pos['qty'] == 0: return
        sell_val = pos['qty'] * price
        cost_val = pos['qty'] * pos['avg_price']
        roi = ((sell_val - cost_val) / cost_val) * 100
        self.cash += sell_val
        
        entry_strategy = pos.get('entry_strategy', '알 수 없음')
        del self.positions[symbol]
        
        self.trade_logs.append({
            "타입": "매도", "종목": symbol, "가격": round(price, 2), 
            "수량": round(pos['qty'], 4), "수익률(%)": round(roi, 2), 
            "진입 전략": entry_strategy, "상세 사유": reason
        })

# 세션 상태 관리
if "trader" not in st.session_state:
    st.session_state.trader = SimulatedTrader()

trader = st.session_state.trader

target_symbols = [
    "이더리움", "SOXL", "KORU", "URAA",
    "팔란티어", "로켓 랩", "슈퍼 마이크로 컴퓨터",
    "알파벳", "애플", "아이온큐", "퀄컴"
]

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
            res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt, config=types.GenerateContentConfig(system_instruction=TRADER_SYSTEM_INSTRUCTION))
            st.info(res.text)
    else:
        st.error("Gemini API 키가 연결되지 않았습니다.")

st.markdown("---")

# ---------------------------------------------------------
# 3. 차트 시각화 UI (네이버 증권 스타일: 초기 확대 + 좌우 드래그 + 휠 스크롤)
# ---------------------------------------------------------
st.subheader("📈 설정 종목 일봉 분석 (최근 90일)")
selected_stock = st.selectbox("종목 선택", target_symbols)
df_selected, vp_top_selected = stock_datas[selected_stock]

curr_price = df_selected['Close'].iloc[-1]
prev_price = df_selected['Close'].iloc[-2]
price_change = ((curr_price - prev_price) / prev_price) * 100

max_idx = df_selected['High'].idxmax()
max_price = df_selected['High'].max()
min_idx = df_selected['Low'].idxmin()
min_price = df_selected['Low'].min()

# 1. 초기 줌 범위 설정 (전체 90일 데이터 중 최근 30일 봉만 확대)
initial_start_date = df_selected.index[-30]
initial_end_date = df_selected.index[-1]

st.metric(
    label=f"{selected_stock} 현재 가격",
    value=f"{int(curr_price):,} 원" if selected_stock == "이더리움" else f"${curr_price:,.2f}",
    delta=f"{price_change:+.2f}%"
)

# (기존) fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

# 👇 아래 코드로 변경 (거래량 창 높이를 기존의 절반 수준인 12%~15%로 축소)
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03, 
    row_heights=[0.88, 0.12]  # 캔들 88%, 거래량 12% 비율로 할당
)

# 캔들스틱 (네이버 증권 스타일: 상승=빨강 #e15241, 하락=파랑 #267af3)
fig.add_trace(go.Candlestick(
    x=df_selected.index,
    open=df_selected['Open'], high=df_selected['High'],
    low=df_selected['Low'], close=df_selected['Close'],
    name="주가",
    increasing_line_color='#e15241', increasing_fillcolor='#e15241',
    decreasing_line_color='#267af3', decreasing_fillcolor='#267af3'
), row=1, col=1)

# 이동평균선 & 볼린저 밴드
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['MA5'], line=dict(color='#34c759', width=1.5), name="MA5"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['MA20'], line=dict(color='#ff9500', width=1.5), name="MA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['BB_Upper'], line=dict(color='#af52de', width=1, dash='dash'), name="BB 상한선"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['BB_Lower'], line=dict(color='#af52de', width=1, dash='dash'), name="BB 하한선"), row=1, col=1)

# 주요 매물대 라인
fig.add_hline(y=vp_top_selected, line_color="#8e8e93", line_dash="dot", annotation_text=f"매물대 상단 ({vp_top_selected:,.2f})", row=1, col=1)

# 최고가 / 최저가 주석 표시
fig.add_annotation(
    x=max_idx, y=max_price, text=f"최고 {max_price:,.2f}",
    showarrow=True, arrowhead=2, arrowcolor="#e15241", ax=0, ay=-25, row=1, col=1
)
fig.add_annotation(
    x=min_idx, y=min_price, text=f"최저 {min_price:,.2f}",
    showarrow=True, arrowhead=2, arrowcolor="#267af3", ax=0, ay=25, row=1, col=1
)

# 거래량 바 차트
vol_colors = ['#e15241' if c >= o else '#267af3' for c, o in zip(df_selected['Close'], df_selected['Open'])]
fig.add_trace(go.Bar(x=df_selected.index, y=df_selected['Volume'], name="거래량", marker_color=vol_colors), row=2, col=1)
fig.add_trace(go.Scatter(x=df_selected.index, y=df_selected['Vol_Avg_90'], line=dict(color='#ff3b30', width=1), name="90일 평균 거래량"), row=2, col=1)

# 2. 레이아웃 및 마우스 드래그(Pan) 모드 설정 (범례 및 말풍선 글자색 추가)
fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=850,
    dragmode='pan',
    margin=dict(l=20, r=20, t=30, b=20),
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#000000'),             # 기본 폰트 검은색
    legend=dict(font=dict(color='#000000')), # 👈 우측 상단 지표 설명(범례) 검은색 강제 지정
    hoverlabel=dict(                         # 👈 마우스 올렸을 때 뜨는 정보창 배경/글자색 지정
        bgcolor='#ffffff',
        font_color='#000000',
        bordercolor='#cccccc'
    ),
    hovermode="x unified"
)

# 3. X/Y축 표시 범위 및 눈금(Tick) 글자색 검은색으로 확정
fig.update_xaxes(
    range=[initial_start_date, initial_end_date],
    showgrid=True, gridwidth=1, gridcolor='#f2f2f7',
    tickfont=dict(color='#000000')  # 👈 X축 날짜 글자색 검은색
)
fig.update_yaxes(
    showgrid=True, gridwidth=1, gridcolor='#f2f2f7', fixedrange=False,
    tickfont=dict(color='#000000')  # 👈 Y축 가격 글자색 검은색
)

# 4. 마우스 휠 확대/축소 옵션
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# ---------------------------------------------------------
# 3-1. 선택 종목 Gemini AI 진단 섹션
# ---------------------------------------------------------
st.markdown(f"#### 🤖 30년차 트레이더의 [{selected_stock}] 기술적 매매 진단")

if st.button(f"🎯 {selected_stock} AI 진단 받아보기", key=f"btn_{selected_stock}"):
    if client:
        with st.spinner(f"30년차 트레이더가 {selected_stock} 차트를 정밀 분석 중입니다..."):
            latest_data = df_selected.iloc[-1]
            pos = trader.positions.get(selected_stock, None)
            pos_info = f"보유 중 (수량: {pos['qty']:.4f}, 평단가: {pos['avg_price']:.2f})" if pos else "현재 미보유"

            prompt = f"""
            [종목명: {selected_stock}]
            - 현재 보유 상태: {pos_info}
            - 현재가: {curr_price:,.2f}
            - 5일 이동평균선: {latest_data['MA5']:,.2f}
            - 20일 이동평균선: {latest_data['MA20']:,.2f}
            - 볼린저 밴드 상한선: {latest_data['BB_Upper']:,.2f} / 하한선: {latest_data['BB_Lower']:,.2f}
            - 금일 거래량: {latest_data['Volume']:,} (90일 평균 거래량: {latest_data['Vol_Avg_90']:,.0f})
            - 주요 매물대 상단 가격: {vp_top_selected:,.2f}

            너는 연 목표수익률 200%를 목표로 하는 30년차 공격적 전문 트레이더이다. 
            위 지표 데이터를 분석하여 아래 형식으로 짧고 명확하게 답변해라:
            
            1. [매매 판단]: 매수(BUY) / 매도(SELL) / 관망(HOLD) 중 택1
            2. [추천 투자 비중]: 전체 계좌 잔고의 % 지정
            3. [트레이딩 사유 & 전략]: 2문장 이내로 핵심 기술적 근거와 대응 전략 제시
            """

            res = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=TRADER_SYSTEM_INSTRUCTION,
                    max_output_tokens=350,
                    temperature=0.4
                )
            )
            st.info(res.text)
    else:
        st.error("Gemini API 키가 연결되지 않았습니다.")

st.markdown("---")

# 4. 매매일지 및 전략별 성과 UI
st.subheader("📑 주식 매매 일지 및 전략 성과 분석")
if trader.trade_logs:
    logs_df = pd.DataFrame(trader.trade_logs)
    
    # 탭을 나누어 매매 일지와 통계를 깔끔하게 분리
    tab1, tab2 = st.tabs(["매매 기록 전체보기", "📊 진입 전략별 승률 통계"])
    
    with tab1:
        st.dataframe(logs_df, use_container_width=True)
        
    with tab2:
        # 매도(청산) 완료된 거래만 필터링하여 승률 계산
        sell_logs = logs_df[logs_df["타입"] == "매도"]
        if not sell_logs.empty:
            summary = sell_logs.groupby("진입 전략").agg(
                총매매횟수=("수익률(%)", "count"),
                익절횟수=("수익률(%)", lambda x: (x > 0).sum()),
                손절횟수=("수익률(%)", lambda x: (x <= 0).sum()),
                평균수익률=("수익률(%)", "mean")
            ).reset_index()
            
            summary["승률(%)"] = (summary["익절횟수"] / summary["총매매횟수"]) * 100
            
            # 보기 좋게 소수점 둘째 자리 정리 및 컬럼 순서 배치
            summary = summary.round(2)
            summary = summary[["진입 전략", "총매매횟수", "승률(%)", "평균수익률", "익절횟수", "손절횟수"]]
            
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("아직 매도(청산)가 완료된 거래가 없어 통계를 낼 수 없습니다.")
else:
    st.write("현재까지 실행된 매매 내역이 없습니다.")
