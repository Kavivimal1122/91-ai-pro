import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

# 1. Page Config for tight mobile display
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Custom CSS for Dialer Grid and Single-Page View
st.markdown("""
    <style>
    /* Remove top margin and padding for mobile */
    .block-container { 
        padding-top: 0.2rem !important; 
        padding-bottom: 0rem !important; 
        padding-left: 0.5rem !important; 
        padding-right: 0.5rem !important; 
    }
    
    /* Result Box styling */
    .pred-box {
        padding: 8px; 
        border-radius: 8px; 
        text-align: center; 
        border: 2px solid white;
        margin-bottom: 2px;
    }

    /* DIALER BUTTONS: Forced Block Style with Big White Numbers */
    div.stButton > button {
        width: 100% !important;
        height: 60px !important;
        border-radius: 5px !important; 
        font-weight: 900 !important;   
        font-size: 28px !important;   
        color: white !important;       
        border: 1px solid white !important;
        margin: 2px 0px !important;
        background-color: #1f1f1f !important;
    }

    /* Mobile specific: ensure the dialer doesn't stretch too wide */
    [data-testid="column"] {
        width: 30% !important;
        flex: 1 1 30% !important;
        min-width: 30% !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Session States
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
if 'stats' not in st.session_state: 
    st.session_state.stats = {"wins": 0, "loss": 0, "win_streak": 0, "loss_streak": 0}
if 'accuracy' not in st.session_state: st.session_state.accuracy = 0

# --- 1. PREDICTION & STREAKS ---
if 'next_num' in st.session_state:
    st.caption(f"🔥 W: {st.session_state.stats['win_streak']} | ❄️ L: {st.session_state.stats['loss_streak']}")
    
    color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
    st.markdown(f"""
        <div class="pred-box" style="background-color: {color};">
            <p style="color: white; margin: 0; font-size: 14px; font-weight: bold;">NEXT: {st.session_state.last_pred_size}</p>
            <h1 style="color: white; margin: 0; font-size: 40px;">{st.session_state.next_num}</h1>
        </div>
    """, unsafe_allow_html=True)

# --- 2. STARTUP LOGIC ---
if st.session_state.ai_model is None:
    file = st.file_uploader("Upload Qus.csv", type="csv")
    if file and st.button("🚀 TRAIN"):
        df = pd.read_csv(file)
        if 'content' in df.columns:
            for i in range(1, 6): df[f'p{i}'] = df['content'].shift(i)
            df = df.dropna()
            model = GradientBoostingClassifier(n_estimators=100).fit(df[['p1','p2','p3','p4','p5']], df['content'])
            tests = random.sample(range(len(df)), 100)
            score = sum(1 for i in tests if model.predict([df.iloc[i][['p1','p2','p3','p4','p5']]])[0] == df.iloc[i]['content'])
            st.session_state.accuracy, st.session_state.ai_model = score, model
            st.rerun()

elif not st.session_state.last_5:
    st.info(f"Training Acc: {st.session_state.accuracy}%")
    init_in = st.text_input("Enter 5 digits", max_chars=5)
    if st.button("CONFIRM START"):
        st.session_state.last_5 = [int(d) for d in init_in]
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

# --- 3. THE DIALER (FORCED 1-2-3 GRID) ---
else:
    new_num = None
    # Use 3 explicit columns for mobile alignment
    c1, c2, c3 = st.columns(3)
    if c1.button("1", key="btn_1"): new_num = 1
    if c2.button("2", key="btn_2"): new_num = 2
    if c3.button("3", key="btn_3"): new_num = 3
    
    c4, c5, c6 = st.columns(3)
    if c4.button("4", key="btn_4"): new_num = 4
    if c5.button("5", key="btn_5"): new_num = 5
    if c6.button("6", key="btn_6"): new_num = 6
    
    c7, c8, c9 = st.columns(3)
    if c7.button("7", key="btn_7"): new_num = 7
    if c8.button("8", key="btn_8"): new_num = 8
    if c9.button("9", key="btn_9"): new_num = 9
    
    c_e1, c0, c_e2 = st.columns(3)
    if c0.button("0", key="btn_0"): new_num = 0

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"
        if actual_size == st.session_state.last_pred_size:
            st.session_state.stats["win_streak"] += 1
            st.session_state.stats["loss_streak"] = 0
            status = "✅ WIN"
        else:
            st.session_state.stats["loss_streak"] += 1
            st.session_state.stats["win_streak"] = 0
            status = "❌ LOSS"
        
        # Point 3: Track and show history
        st.session_state.history.insert(0, {"#": new_num, "Result": status})
        if len(st.session_state.history) > 20: st.session_state.history.pop()
        
        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

    # --- 4. HISTORY (LAST 20) ---
    if st.session_state.history:
        st.write("---")
        st.subheader("📜 History (Last 20)")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

    if st.button("RESET ALL", key="reset"):
        st.session_state.clear()
        st.rerun()
