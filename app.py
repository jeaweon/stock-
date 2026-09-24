import os
import json
import requests  # 🚨 새로 추가된 부분!
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
    """원/달러 환율 120일치 데이터를 가져오는 함수"""
    fx_df = yf.download("KRW=X", period="120d", interval="1d")
    if isinstance(fx_df.columns, pd.MultiIndex):
        fx_df.columns = fx_df.columns.droplevel(1)
        
    if fx_df.index.tz is not None:
        fx_df.index = fx_df.index.tz_localize(None)
        
    # 빈칸 처리 유연화
    fx_df = fx_df.dropna(how='all') 
    fx_df = fx_df.ffill()
    
    return fx_df['Close']

@st.cache_data
def fetch_stock_data(symbol_name):
    symbol_map = {
        "이더리움": "ETH-USD",
        "SOXL": "SOXL",
        "KORU": "KORU",
        "URAA": "URAA",
        "팔란티어": "PLTR",
        "로켓 랩": "RKLB",
        "슈퍼 마이크로 컴퓨터": "SMCI",
        "알파벳": "GOOGL",
        "애플": "AAPL",
        "아이온큐": "IONQ",
        "퀄컴": "QCOM"
    }
    ticker = symbol_map.get(symbol_name, symbol_name)
    
    # 1. 정규장 일봉 데이터 다운로드
    df = yf.download(ticker, period="120d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # 🚨 [핵심 추가] 프리장, 애프터장, 데이장(장외) 실시간 데이터 반영
    if symbol_name != "이더리움":  # 코인은 장외 개념이 없으므로 제외
        try:
            # prepost=True 옵션으로 장외 시간을 포함한 가장 최신 1분봉 데이터를 불러옴
            ext_df = yf.download(ticker, period="1d", interval="1m", prepost=True, progress=False)
            if isinstance(ext_df.columns, pd.MultiIndex):
                ext_df.columns = ext_df.columns.droplevel(1)
                
            if not ext_df.empty and not ext_df['Close'].dropna().empty:
                # 장외 거래가 포함된 가장 마지막 실시간 체결 가격
                latest_ext_price = float(ext_df['Close'].dropna().iloc[-1])
                
                # 오늘(또는 가장 최근 일봉)의 종가를 장외 가격으로 실시간 업데이트
                df.iloc[-1, df.columns.get_loc('Close')] = latest_ext_price
                
                # 장외 가격이 급등/급락하여 기존 정규장 고가/저가를 돌파했다면 캔들의 꼬리(High/Low)도 갱신
                if latest_ext_price > df.iloc[-1]['High']:
                    df.iloc[-1, df.columns.get_loc('High')] = latest_ext_price
                if latest_ext_price < df.iloc[-1]['Low']:
                    df.iloc[-1, df.columns.get_loc('Low')] = latest_ext_price
        except Exception as e:
            pass # 일시적인 통신 에러 발생 시 무시하고 기존 정규장 데이터 유지
            
    # 빈칸 처리
    df = df.dropna(how='all')  
    df = df.ffill()            
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 2. 환율 데이터 가져와서 날짜별로 매칭
    fx_series = fetch_usd_krw()
    df = df.join(fx_series.rename("FX_Rate"), how="left")
    df['FX_Rate'] = df['FX_Rate'].ffill().bfill() 

    # 3. 달러(USD) 가격 * 원/달러 환율 = 실제 원화(KRW) 가격으로 변환
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
        # 주말 휴장을 고려해 넉넉히 5d로 설정하여 0.00 오류 방지
        t_data = yf.Ticker(sym).history(period="5d")
        if len(t_data) >= 2:
            close = float(t_data['Close'].iloc[-1])
            prev = float(t_data['Close'].iloc[-2])
            data[name] = {"price": close, "change": ((close - prev) / prev) * 100}
        else:
            data[name] = {"price": 0.0, "change": 0.0}
    return data

# ---------------------------------------------------------
# 가상 매매 엔진 Class (Gist 클라우드 영구 보존 버전)
# ---------------------------------------------------------
class SimulatedTrader:
    def __init__(self, initial_balance=20000000, save_file="trader_state.json"):
        self.save_file = save_file
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions = {}
        self.trade_logs = []
        
        # 🚨 Streamlit Secrets에서 Gist 연동 키 불러오기
        self.gist_id = st.secrets.get("GIST_ID", "")
        self.github_token = st.secrets.get("GITHUB_TOKEN", "")
        
        self._load_from_file()

    def _save_to_file(self):
        state = {
            "cash": self.cash,
            "initial_balance": self.initial_balance,
            "positions": self.positions,
            "trade_logs": self.trade_logs
        }
        
        # 1. 로컬 저장 (예비용)
        with open(self.save_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
            
        # 2. 클라우드(Gist) 실시간 백업
        if self.gist_id and self.github_token:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            data = {
                "files": {
                    self.save_file: {
                        "content": json.dumps(state, ensure_ascii=False, indent=4)
                    }
                }
            }
            try:
                requests.patch(f"https://api.github.com/gists/{self.gist_id}", headers=headers, json=data)
            except Exception as e:
                print(f"Gist 백업 에러: {e}")

    def _load_from_file(self):
        state = None
        
        # 1. 서버 재부팅 시 가장 최신인 클라우드(Gist)에서 우선 로드
        if self.gist_id and self.github_token:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            try:
                res = requests.get(f"https://api.github.com/gists/{self.gist_id}", headers=headers)
                if res.status_code == 200:
                    gist_data = res.json()
                    file_content = gist_data["files"].get(self.save_file, {}).get("content")
                    if file_content:
                        state = json.loads(file_content)
            except Exception as e:
                print(f"Gist 로드 에러: {e}")
                
        # 2. 클라우드 로드 실패 시, 로컬에서 시도
        if state is None and os.path.exists(self.save_file):
            with open(self.save_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                
        # 3. 데이터 적용
        if state:
            self.cash = state.get("cash", self.initial_balance)
            self.initial_balance = state.get("initial_balance", self.initial_balance)
            self.positions = state.get("positions", {})
            self.trade_logs = state.get("trade_logs", [])

    # ... (아래 execute_strategy, _buy, _sell 함수는 기존과 동일하게 유지) ...
    # ⚠️ 방어 로직과 날짜 기록(trade_date) 로직 그대로 두시면 됩니다!

    def execute_strategy(self, symbol, df, vp_top):
        latest = df.iloc[-1]
        
        # 현재 캔들의 날짜를 문자열(YYYY-MM-DD)로 추출
        current_date_str = str(latest.name).split()[0]
        
        price = latest['Close']
        ma5, ma20 = latest['MA5'], latest['MA20']
        vol, vol_avg_90 = latest['Volume'], latest['Vol_Avg_90']
        bb_lower = latest['BB_Lower']
        
        pos = self.positions.get(symbol, {'qty': 0, 'avg_price': 0.0, 'bb_stage': 0})
        
        # [핵심 방어 로직] 오늘 이미 매수/매도를 진행했다면 더 이상 거래하지 않음
        if pos.get('last_trade_date') == current_date_str:
            return
            
        # 볼린저 밴드 하한선 물타기 로직
        if latest['Low'] <= bb_lower:
            stage = pos.get('bb_stage', 0)
            if stage == 0:
                self._buy(symbol, price, self.cash * 0.30, "BB 1차 하한선 (잔고 30% / 목표익절 +3%)", 1, current_date_str)
                return
            elif stage == 1:
                self._buy(symbol, price, self.cash * 0.50, "BB 2차 하한선 (잔고 50% / 목표익절 +1%)", 2, current_date_str)
                return
            elif stage >= 2:
                self._buy(symbol, price, self.cash * 0.50, "BB 3차 하한선 (잔고 50% / 목표익절 0%)", 3, current_date_str)
                return
        
        # 볼린저 밴드 목표 달성 익절
        if pos.get('qty', 0) > 0 and pos.get('bb_stage', 0) > 0:
            roi = ((price - pos.get('avg_price', 1)) / pos.get('avg_price', 1)) * 100
            target = {1: 3.0, 2: 1.0, 3: 0.0}.get(pos.get('bb_stage', 0), 0.0)
            if roi >= target:
                self._sell(symbol, price, f"BB {pos.get('bb_stage', 0)}단계 목표 수익률({target}%) 달성 매도", current_date_str)
                return

        # VCP 및 매물대 풀백 등 신규 매수 조건
        prev_bb_width = df['BB_Width'].iloc[-2]
        cond_vcp = (prev_bb_width < 0.10) and (vol >= vol_avg_90 * 3.0) and (price > latest['BB_Upper'])

        is_yangbong = latest['Close'] > latest['Open']
        touched_support = latest['Low'] <= (vp_top * 1.02)
        cond_pullback = is_yangbong and touched_support and (price > vp_top) and (ma5 > ma20)

        cond1 = (price > vp_top) and (ma5 > ma20)
        cond2 = (vol > vol_avg_90 * 1.5) and (ma5 > ma20)
        
        if pos.get('qty', 0) == 0:
            if cond_vcp:
                self._buy(symbol, price, self.cash * 0.40, "🔥 VCP 돌파: BB 극도 수축 후 3배 거래량 폭발", 0, current_date_str)
            elif cond_pullback:
                self._buy(symbol, price, self.cash * 0.35, "🎯 매물대 풀백: 핵심 저항 지지 후 첫 양봉 반등", 0, current_date_str)
            elif cond1:
                self._buy(symbol, price, self.cash * 0.20, "매물대 돌파 및 MA5 > MA20", 0, current_date_str)
            elif cond2:
                self._buy(symbol, price, self.cash * 0.20, "거래량 1.5배 돌파 및 MA5 > MA20", 0, current_date_str)
                
        elif pos.get('qty', 0) > 0 and pos.get('bb_stage', 0) == 0:
            if ma20 > ma5:
                self._sell(symbol, price, "MA20 > MA5 데드크로스 매도", current_date_str)

    def _buy(self, symbol, price, amount, reason, bb_stage, trade_date):
        if amount < 10000: return
        qty = amount / price
        pos = self.positions.get(symbol, {'qty': 0, 'avg_price': 0.0, 'bb_stage': 0})
        
        entry_strategy = reason if pos.get('qty', 0) == 0 else pos.get('entry_strategy', reason)
        total_qty = pos.get('qty', 0) + qty
        total_cost = (pos.get('qty', 0) * pos.get('avg_price', 0.0)) + amount
        
        self.cash -= amount
        
        self.positions[symbol] = {
            'qty': total_qty, 
            'avg_price': total_cost / total_qty, 
            'bb_stage': bb_stage, 
            'entry_strategy': entry_strategy,
            'last_trade_date': trade_date 
        }
        
        self.trade_logs.append({
            "일자": trade_date,
            "타입": "매수", "종목": symbol, "가격": round(price, 2), 
            "수량": round(qty, 4), "수익률(%)": 0.0, 
            "진입 전략": entry_strategy, "상세 사유": reason
        })
        self._save_to_file()

    def _sell(self, symbol, price, reason, trade_date):
        pos = self.positions.get(symbol)
        if not pos or pos.get('qty', 0) == 0: return
        
        sell_val = pos.get('qty', 0) * price
        cost_val = pos.get('qty', 0) * pos.get('avg_price', 0.0)
        roi = ((sell_val - cost_val) / cost_val) * 100
        self.cash += sell_val
        
        entry_strategy = pos.get('entry_strategy', '알 수 없음')
        del self.positions[symbol]
        
        self.trade_logs.append({
            "일자": trade_date,
            "타입": "매도", "종목": symbol, "가격": round(price, 2), 
            "수량": round(pos.get('qty', 0), 4), "수익률(%)": round(roi, 2), 
            "진입 전략": entry_strategy, "상세 사유": reason
        })
        self._save_to_file()

# ---------------------------------------------------------
# 세션 상태 관리 및 완벽 초기화 (Gist 클라우드 동기화 포함)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 시스템 관리")
    
    # 1. 주가 데이터 수동 갱신 버튼
    if st.button("📈 최신 주가 데이터 불러오기"):
        # 저장된 데이터 캐시(기억)를 모두 강제 삭제
        fetch_usd_krw.clear()
        fetch_stock_data.clear()
        fetch_market_indices.clear()
        st.success("최신 주가 데이터를 성공적으로 불러왔습니다!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 💾 데이터 백업 및 복구")
    st.caption("수동 백업용: 내 컴퓨터에 저장하고 다시 불러오기")

    # 2. 백업 파일 다운로드
    if os.path.exists("trader_state.json"):
        with open("trader_state.json", "r", encoding="utf-8") as f:
            save_data = f.read()
        st.download_button(
            label="📥 현재 계좌 상태 다운로드 (수동 백업)",
            data=save_data,
            file_name="trader_state.json",
            mime="application/json",
            help="클라우드 동기화와 별개로 내 컴퓨터에 수동으로 백업할 때 사용하세요."
        )

    # 3. 백업 파일 업로드(복구)
    uploaded_file = st.file_uploader("📤 수동 백업 파일 복구", type=["json"])
    if uploaded_file is not None:
        if st.button("수동 데이터 복구 실행"):
            # 업로드한 파일을 로컬에 덮어쓰기
            file_content_bytes = uploaded_file.getvalue()
            with open("trader_state.json", "wb") as f:
                f.write(file_content_bytes)
                
            # 복구한 데이터를 클라우드 Gist에도 즉시 반영
            gist_id = st.secrets.get("GIST_ID", "")
            github_token = st.secrets.get("GITHUB_TOKEN", "")
            if gist_id and github_token:
                try:
                    # bytes를 json 문자열로 디코딩
                    decoded_str = file_content_bytes.decode('utf-8')
                    # 올바른 json 형태인지 확인(파싱) 후 다시 문자열로 포맷팅
                    parsed_json = json.loads(decoded_str)
                    
                    headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}
                    data = {"files": {"trader_state.json": {"content": json.dumps(parsed_json, ensure_ascii=False, indent=4)}}}
                    requests.patch(f"https://api.github.com/gists/{gist_id}", headers=headers, json=data)
                except Exception as e:
                    st.error(f"클라우드 동기화 복구 실패: {e}")
                    
            st.session_state.clear()
            st.rerun()

    st.markdown("---")
    
    # 4. 계좌 완전 초기화 버튼 (로컬 + 클라우드 리셋)
    if st.button("🔄 모의투자 계좌 초기화 (완전 리셋)"):
        # 로컬 파일 삭제
        if os.path.exists("trader_state.json"):
            os.remove("trader_state.json")
            
        # 클라우드 Gist 데이터도 함께 초기화
        gist_id = st.secrets.get("GIST_ID", "")
        github_token = st.secrets.get("GITHUB_TOKEN", "")
        if gist_id and github_token:
            empty_state = {
                "cash": 20000000, "initial_balance": 20000000, 
                "positions": {}, "trade_logs": []
            }
            headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}
            data = {"files": {"trader_state.json": {"content": json.dumps(empty_state)}}}
            requests.patch(f"https://api.github.com/gists/{gist_id}", headers=headers, json=data)

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
            try:
                prompt = f"""
                지수 현황: {indices}

                오늘 장 마감 후 시장 요인을
                3줄로 날카롭게 요약해라.
                """

                res = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=TRADER_SYSTEM_INSTRUCTION
                    )
                )

                st.info(res.text)

            except Exception as e:
                st.error("Gemini API 호출에 실패했습니다.")
                st.exception(e)
    else:
        st.error("Gemini API 키가 연결되지 않았습니다.")

st.markdown("---")

# ---------------------------------------------------------
# 3. 차트 시각화 UI 
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

initial_start_date = df_selected.index[-30]
initial_end_date = df_selected.index[-1]

st.metric(
    label=f"{selected_stock} 현재 가격",
    value=f"{int(curr_price):,} 원",
    delta=f"{price_change:+.2f}%"
)

fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03, 
    row_heights=[0.88, 0.12]  
)

# 캔들스틱 
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

# 최고가 / 최저가 주석 
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

# 레이아웃 및 디자인
fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=850,
    dragmode='pan',
    margin=dict(l=20, r=20, t=30, b=20),
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#000000'),             
    legend=dict(font=dict(color='#000000')), 
    hoverlabel=dict(                         
        bgcolor='#ffffff',
        font_color='#000000',
        bordercolor='#cccccc'
    ),
    hovermode="x unified"
)

# 3. X/Y축 표시 범위 및 눈금(Tick) 글자색 검은색으로 확정
# 코인(이더리움)은 주말 장이 열리므로 그대로 두고, 일반 주식만 주말 공백을 제거함
if selected_stock == "이더리움":
    fig.update_xaxes(
        range=[initial_start_date, initial_end_date],
        showgrid=True, gridwidth=1, gridcolor='#f2f2f7',
        tickfont=dict(color='#000000')
    )
else:
    fig.update_xaxes(
        range=[initial_start_date, initial_end_date],
        showgrid=True, gridwidth=1, gridcolor='#f2f2f7',
        tickfont=dict(color='#000000'),
        rangebreaks=[
            dict(bounds=["sat", "mon"])  # 토요일부터 월요일 전까지 잘라내기
        ]
    )

# Y축 설정 (이 부분이 지워지면 에러가 납니다!)
fig.update_yaxes(
    showgrid=True, gridwidth=1, gridcolor='#f2f2f7', fixedrange=False,
    tickfont=dict(color='#000000')
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
            - 현재가: {int(curr_price):,} 원
            - 5일 이동평균선: {int(latest_data['MA5']):,} 원
            - 20일 이동평균선: {int(latest_data['MA20']):,} 원
            - 볼린저 밴드 상한선: {int(latest_data['BB_Upper']):,} 원 / 하한선: {int(latest_data['BB_Lower']):,} 원
            - 금일 거래량: {latest_data['Volume']:,} (90일 평균 거래량: {latest_data['Vol_Avg_90']:,.0f})
            - 주요 매물대 상단 가격: {int(vp_top_selected):,} 원

            너는 연 목표수익률 200%를 목표로 하는 30년차 공격적 전문 트레이더이다. 
            위 지표 데이터를 분석하여 아래 형식으로 짧고 명확하게 답변해라:
            
            1. [매매 판단]: 매수(BUY) / 매도(SELL) / 관망(HOLD) 중 택1
            2. [추천 투자 비중]: 전체 계좌 잔고의 % 지정
            3. [트레이딩 사유 & 전략]: 2문장 이내로 핵심 기술적 근거와 대응 전략 제시
            """

            res = client.models.generate_content(
                model="gemini-3.6-flash",
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
    
    tab1, tab2 = st.tabs(["매매 기록 전체보기", "📊 진입 전략별 승률 통계"])
    
    with tab1:
        st.dataframe(logs_df, use_container_width=True)
        
    with tab2:
        sell_logs = logs_df[logs_df["타입"] == "매도"]
        if not sell_logs.empty:
            summary = sell_logs.groupby("진입 전략").agg(
                총매매횟수=("수익률(%)", "count"),
                익절횟수=("수익률(%)", lambda x: (x > 0).sum()),
                손절횟수=("수익률(%)", lambda x: (x <= 0).sum()),
                평균수익률=("수익률(%)", "mean")
            ).reset_index()
            
            summary["승률(%)"] = (summary["익절횟수"] / summary["총매매횟수"]) * 100
            
            summary = summary.round(2)
            summary = summary[["진입 전략", "총매매횟수", "승률(%)", "평균수익률", "익절횟수", "손절횟수"]]
            
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("아직 매도(청산)가 완료된 거래가 없어 통계를 낼 수 없습니다.")
else:
    st.write("현재까지 실행된 매매 내역이 없습니다.")
