import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

# 1. Compact Page Config
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Custom CSS for Solid Blocks and Zero Padding
st.markdown("""
    <style>
    /* Force everything to fit on one mobile screen */
    .block-container { padding-top: 0.5rem; padding-bottom: 0rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    
    /* Result Box */
    .pred-box {
        padding: 10px; 
        border-radius: 8px; 
        text-align: center; 
        border: 2px solid white;
        margin-bottom: 5px;
    }

    /* Solid Block Buttons with HUGE White Numbers */
    div.stButton > button {
        width: 100% !important;
        height: 65px !important;
        border-radius: 5px !important; 
        font-weight: 900 !important;   
        font-size: 30px !important;   
        color: white !important;       
        border: 1px solid white !important;
        margin: 0px !important;
    }
    
    /* Block Color: Green (0-4) with White Letters */
    div.stButton > button[key^="btn_0"], div.stButton > button[key^="btn_1"], 
    div.stButton > button[key^="btn_2"], div.stButton > button[key^="btn_3"], 
    div.stButton > button[key^="btn_4"] { 
        background-color: #28a745 !important; 
        color: white !important;
    }
    
    /* Block Color: Red (5-9) with White Letters */
    div.stButton > button[key^="btn_5"], div.stButton > button[key^="btn_6"], 
    div.stButton > button[key^="btn_7"], div.stButton > button[key^="btn_8"], 
    div.stButton > button[key^="btn_9"] { 
        background-color: #dc3545 !important; 
        color: white !important;
    }

    /* Hide Streamlit elements to save space */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Session States
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
# Added streak tracking
if 'stats' not in st.session_state: 
    st.session_state.stats = {"wins": 0, "loss": 0, "win_streak": 0, "loss_streak": 0}

# --- 1. PREDICTION & STREAKS (TOP) ---
if 'next_num' in st.session_state:
    # Display Streaks at the very top
    st.write(f"🔥 Win Streak: {st.session_state.stats['win_streak']} | ❄️ Loss Streak: {st.session_state.stats['loss_streak']}")
    
    color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
    st.markdown(f"""
        <div class="pred-box" style="background-color: {color};">
            <h2 style="color: white; margin: 0; font-size: 18px;">NEXT: {st.session_state.last_pred_size}</h2>
            <h1 style="color: white; margin: 0; font-size: 45px;">{st.session_state.next_num}</h1>
        </div>
    """, unsafe_allow_html=True)

# --- 2. STARTUP LOGIC ---
if st.session_state.ai_model is None:
    file = st.file_uploader("Upload Qus.csv", type="csv")
    if file and st.button("🚀 TRAIN"):
        df = pd.read_csv(file)
        for i in range(1, 6): df[f'p{i}'] = df['content'].shift(i)
        df = df.dropna()
        model = GradientBoostingClassifier(n_estimators=100).fit(df[['p1','p2','p3','p4','p5']], df['content'])
        st.session_state.ai_model = model
        st.rerun()
elif not st.session_state.last_5:
    init_in = st.text_input("Enter 5 digits (e.g. 35125)", max_chars=5)
    if st.button("CONFIRM"):
        st.session_state.last_5 = [int(d) for d in init_in]
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

# --- 3. THE DIALER (CENTER - NO SCROLL) ---
else:
    new_num = None
    # 3x3 Grid
    for row in [[1, 2, 3], [4, 5, 6], [7, 8, 9]]:
        cols = st.columns(3)
        for i, n in enumerate(row):
            if cols[i].button(str(n), key=f"btn_{n}"): new_num = n
            
    # Bottom Row for 0
    c1, c2, c3 = st.columns(3)
    if c2.button("0", key="btn_0"): new_num = 0

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"
        
        # Win/Loss and Streak Logic
        if actual_size == st.session_state.last_pred_size:
            st.session_state.stats["wins"] += 1
            st.session_state.stats["win_streak"] += 1
            st.session_state.stats["loss_streak"] = 0
            res_status = "✅ WIN"
        else:
            st.session_state.stats["loss"] += 1
            st.session_state.stats["loss_streak"] += 1
            st.session_state.stats["win_streak"] = 0
            res_status = "❌ LOSS"
        
        # Save to History (stores last 20)
        st.session_state.history.insert(0, {"Number": new_num, "Result": res_status})
        if len(st.session_state.history) > 20:
            st.session_state.history.pop()
        
        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

    # --- 4. HISTORY TABLE (LAST 20) ---
    if st.session_state.history:
        st.write("---")
        st.subheader("📜 Last 20 History")
        st.table(pd.DataFrame(st.session_state.history))

    # Reset at bottom
    if st.button("RESET", key="reset_app"):
        st.session_state.clear()
        st.rerun()
