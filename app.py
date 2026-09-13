import os
import json
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

st.caption("🔄 데이터 갱신 주기: 수동 업데이트 (사이드바의 '최신 주가 데이터 불러오기' 버튼 클릭)")

# API 키 가져오기
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = None
    st.warning("⚠️ Secrets에 GEMINI_API_KEY가 설정되지 않았습니다.")

TRADER_SYSTEM_INSTRUCTION = """
너는 30년차 전문 월스트리트 트레이더이자 거시경제 전문가야. 주식을 볼 때 일봉, 거래량, 매물대, 볼린저 밴드, 이동 평균선 전체, AD라인, 미국의 통화정책, 경제 뉴스를 보면서 투자를 해.
또 단순한 차트 분석을 넘어, 미국의 통화정책(Fed 금리 결정), 인플레이션 지표, 글로벌 매크로 환경, 그리고 종목별 최신 핵심 뉴스를 완벽하게 융합하여 분석하는 능력을 갖추고 있어.
목표 수익률은 1년에 200%인 공격적인 투자자이며, 정보의 홍수 속에서 노이즈를 걸러내고 핵심 모멘텀만 집어내는 날카롭고 단호한 톤으로 리포트를 작성해라.
"""

# ---------------------------------------------------------
# 데이터 수집 및 계산 함수
# ---------------------------------------------------------
@st.cache_data
def fetch_usd_krw():
    """원/달러 환율 데이터를 가져오는 함수"""
    fx_df = yf.download("KRW=X", period="120d", interval="1d")
    if isinstance(fx_df.columns, pd.MultiIndex):
        fx_df.columns = fx_df.columns.droplevel(1)
        
    # 날짜 인덱스의 시간대(timezone) 제거 (주식/코인 데이터와 병합 시 오류 방지)
    if fx_df.index.tz is not None:
        fx_df.index = fx_df.index.tz_localize(None)
        
    return fx_df['Close']

@st.cache_data
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
    
    # 1. 주식/코인 데이터 다운로드
    df = yf.download(ticker, period="120d", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna()
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 2. 환율 데이터 가져와서 날짜별로 매칭 (주말 코인장은 금요일 환율로 채움)
    fx_series = fetch_usd_krw()
    df = df.join(fx_series.rename("FX_Rate"), how="left")
    df['FX_Rate'] = df['FX_Rate'].ffill().bfill() 

    # 3. 핵심: 달러(USD) 가격 * 원/달러 환율 = 실제 원화(KRW) 가격으로 변환
    # 이더리움(약 3000달러 * 1350원 = 약 400만 원)으로 정상 변환됩니다.
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col] * df['FX_Rate']

    # 4. 원화로 변환된 가격을 바탕으로 보조 지표 계산
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
@st.cache_data
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
# 가상 매매 엔진 Class (영구 보존 버전)
# ---------------------------------------------------------
class SimulatedTrader:
    def __init__(self, initial_balance=20000000, save_file="trader_state.json"):
        self.save_file = save_file
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions = {}
        self.trade_logs = []
        
        # 객체가 생성될 때 저장된 파일이 있으면 불러오기
        self._load_from_file()

    def _save_to_file(self):
        """현재 계좌 상태와 매매 일지를 JSON 파일로 저장"""
        state = {
            "cash": self.cash,
            "initial_balance": self.initial_balance,
            "positions": self.positions,
            "trade_logs": self.trade_logs
        }
        with open(self.save_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

    def _load_from_file(self):
        """저장된 JSON 파일에서 계좌 상태와 매매 일지 불러오기"""
        if os.path.exists(self.save_file):
            with open(self.save_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                self.cash = state.get("cash", self.initial_balance)
                self.initial_balance = state.get("initial_balance", self.initial_balance)
                self.positions = state.get("positions", {})
                self.trade_logs = state.get("trade_logs", [])

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

        # VCP 및 매물대 풀백 등 신규 매수 조건
        prev_bb_width = df['BB_Width'].iloc[-2]
        cond_vcp = (prev_bb_width < 0.10) and (vol >= vol_avg_90 * 3.0) and (price > latest['BB_Upper'])

        is_yangbong = latest['Close'] > latest['Open']
        touched_support = latest['Low'] <= (vp_top * 1.02)
        cond_pullback = is_yangbong and touched_support and (price > vp_top) and (ma5 > ma20)

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

    def _buy(self, symbol, price, amount, reason, bb_stage):
        if amount < 10000: return
        qty = amount / price
        pos = self.positions.get(symbol, {'qty': 0, 'avg_price': 0.0, 'bb_stage': 0})
        
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
        self._save_to_file()  # 상태 변경 후 즉시 저장

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
        self._save_to_file()  # 상태 변경 후 즉시 저장

# 세션 상태 관리
# ---------------------------------------------------------
# 세션 상태 관리 및 완벽 초기화 (데이터 수동 갱신 포함)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 시스템 관리")
    
    # 1. 주가 데이터 수동 갱신 버튼 (추가됨)
    if st.button("📈 최신 주가 데이터 불러오기"):
        # 저장된 데이터 캐시(기억)를 모두 강제 삭제
        fetch_usd_krw.clear()
        fetch_stock_data.clear()
        fetch_market_indices.clear()
        st.success("최신 주가 데이터를 성공적으로 불러왔습니다!")
        st.rerun()

    st.markdown("---")
    
    # 2. 계좌 완전 초기화 버튼 (기존 유지)
    if st.button("🔄 모의투자 계좌 초기화 (완전 리셋)"):
        if os.path.exists("trader_state.json"):
            os.remove("trader_state.json")
        st.session_state.clear()
        st.rerun()

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

# ---------------------------------------------------------
# 2. 시장 지수 및 Gemini 시황 분석 UI
# ---------------------------------------------------------
st.subheader("🌐 대표 시장 지수")
indices = fetch_market_indices()
idx_cols = st.columns(len(indices))

# 지수(나스닥 등)와 환율/이더리움 화면에 렌더링
for idx, (name, val) in enumerate(indices.items()):
    idx_cols[idx].metric(label=name, value=f"{val['price']:,.2f}", delta=f"{val['change']:.2f}%")

st.markdown("---")
st.markdown("### 📊 장 마감 제미나이 매크로 시황 분석")
if st.button("제미나이 AI 지수 & 매크로 분석 실행"):
    if client:
        with st.spinner("30년차 트레이더가 최신 글로벌 매크로 지표를 분석 중입니다..."):
            try:
                prompt = f"""
                현재 주요 시장 지수 현황: {indices}

                위 지수 데이터를 바탕으로 오늘 시장에 영향을 미친 핵심 요인들을 분석해라.
                아래 3가지 목차로 나누어 전문 트레이더의 시각으로 작성해라:

                1. 🌐 글로벌 매크로 현황
                2. 📰 오늘의 핵심 마켓 뷰
                3. 💡 트레이더의 인사이트
                """

                res = client.models.generate_content(
                    model="gemini-3.6-flash", # 질문자님의 기존 모델명 유지
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=TRADER_SYSTEM_INSTRUCTION,
                        temperature=0.5,
                        max_output_tokens=500 # 1500에서 500으로 축소하여 자원 절약
                        # tools 부분 삭제하여 API 과부하 방지
                    )
                )
                st.info(res.text)

            except Exception as e:
                st.error("Gemini API 호출에 실패했습니다.")
                st.exception(e)
    else:
        st.error("Gemini API 키가 연결되지 않았습니다.")

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
    value=f"{int(curr_price):,} 원",
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
if st.button(f"🎯 {selected_stock} 심층 AI 진단 받아보기", key=f"btn_{selected_stock}"):
    if client:
        with st.spinner(f"최신 뉴스와 차트를 융합하여 {selected_stock}를 정밀 분석 중입니다... (약 10~20초 소요)"):
            latest_data = df_selected.iloc[-1]
            pos = trader.positions.get(selected_stock, None)
            pos_info = f"보유 중 (수량: {pos['qty']:.4f}, 평단가: {int(pos['avg_price']):,} 원)" if pos else "현재 미보유"

            prompt = f"""
            [분석 대상 종목: {selected_stock}]
            - 현재 보유 상태: {pos_info}
            - 기술적 데이터:
              * 현재가: {int(curr_price):,} 원
              * 5일 이동평균선: {int(latest_data['MA5']):,} 원
              * 20일 이동평균선: {int(latest_data['MA20']):,} 원
              * 볼린저 밴드 상한: {int(latest_data['BB_Upper']):,} 원 / 하한: {int(latest_data['BB_Lower']):,} 원
              * 금일 거래량: {latest_data['Volume']:,} (90일 평균: {latest_data['Vol_Avg_90']:,.0f})
              * 주요 매물대 상단 가격: {int(vp_top_selected):,} 원

            너는 연 목표수익률 200%의 30년차 월스트리트 전문 트레이더다.
            반드시 '구글 검색'을 사용하여 {selected_stock}와 관련된 가장 최근의 뉴스, 실적 발표, 파이프라인, 거시경제(금리 등) 영향을 파악한 뒤, 
            제공된 기술적 차트 지표와 결합하여 아래의 리포트 형식으로 상세히 분석해라.

            [리포트 양식]
            1. 📰 기본적 분석 (뉴스 & 매크로 모멘텀)
               - 최근 발생한 핵심 호재/악재 뉴스 요약
               - 금리 등 거시경제가 해당 종목에 미치는 영향
            2. 📈 기술적 분석 (차트 및 수급)
               - 이동평균선, 볼린저 밴드, 매물대 및 거래량을 통한 현재 주가 위치 진단
            3. ⚔️ 매매 판단 (BUY / SELL / HOLD) 및 추천 비중
               - 명확한 포지션 제시 및 전체 계좌 대비 추천 비중(%)
            4. 🎯 단기 대응 전략
               - 목표가, 손절가 및 구체적인 시나리오 기반의 행동 지침
            """

            res = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=TRADER_SYSTEM_INSTRUCTION,
                    max_output_tokens=1500, # 상세한 답변을 위해 350 -> 1500으로 대폭 상향
                    temperature=0.5,
                    # 🔥 핵심: 구글 검색을 통해 해당 종목 최신 기사 크롤링
                    tools=[{"google_search": {}}]
                )
            )
            
            # 분석 결과 출력
            st.markdown(res.text)
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
