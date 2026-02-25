import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

# 1. Compact Page Config for Mobile
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Custom CSS to force single-page view on mobile
st.markdown("""
    <style>
    /* Remove huge gaps at the top and sides */
    .block-container { 
        padding-top: 0rem !important; 
        padding-bottom: 0rem !important; 
        padding-left: 0.2rem !important; 
        padding-right: 0.2rem !important; 
    }
    
    /* Shrink the prediction box to save space */
    .pred-box {
        padding: 5px; 
        border-radius: 8px; 
        text-align: center; 
        border: 2px solid white;
        margin-bottom: 2px;
    }

    /* DIALER BUTTONS: Optimized for mobile touch without scrolling */
    div.stButton > button {
        width: 100% !important;
        height: 55px !important;
        border-radius: 6px !important; 
        font-weight: 900 !important;   
        font-size: 24px !important;   
        color: white !important;       
        border: none !important;
        margin: 2px 0px !important;
        background-color: #1f1f1f !important;
    }

    /* Make the table smaller */
    .stTable { font-size: 12px !important; }
    
    /* Hide Streamlit UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Reduce vertical spacing between elements */
    .element-container { margin-bottom: -10px !important; }
    </style>
""", unsafe_allow_html=True)

# Session States
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
if 'stats' not in st.session_state: 
    st.session_state.stats = {"wins": 0, "loss": 0, "win_streak": 0, "loss_streak": 0}
if 'accuracy' not in st.session_state: st.session_state.accuracy = 0

# --- 1. PREDICTION & STREAKS (TOP) ---
if 'next_num' in st.session_state:
    # Compact Streak Text
    st.caption(f"🔥 W: {st.session_state.stats['win_streak']} | ❄️ L: {st.session_state.stats['loss_streak']}")
    
    color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
    st.markdown(f"""
        <div class="pred-box" style="background-color: {color};">
            <p style="color: white; margin: 0; font-size: 14px; font-weight: bold;">NEXT: {st.session_state.last_pred_size}</p>
            <h1 style="color: white; margin: 0; font-size: 35px;">{st.session_state.next_num}</h1>
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
        tests = random.sample(range(len(df)), 100)
        score = sum(1 for i in tests if model.predict([df.iloc[i][['p1','p2','p3','p4','p5']]])[0] == df.iloc[i]['content'])
        st.session_state.accuracy = score
        st.session_state.ai_model = model
        st.rerun()

elif not st.session_state.last_5:
    st.info(f"Acc: {st.session_state.accuracy}%")
    init_in = st.text_input("Enter 5 digits", max_chars=5)
    if st.button("CONFIRM"):
        st.session_state.last_5 = [int(d) for d in init_in]
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

# --- 3. THE DIALER ---
else:
    new_num = None
    # 3x3 Grid
    for row in [[1, 2, 3], [4, 5, 6], [7, 8, 9]]:
        cols = st.columns(3)
        for i, n in enumerate(row):
            if cols[i].button(str(n), key=f"btn_{n}"): new_num = n
            
    c1, c2, c3 = st.columns(3)
    if c2.button("0", key="btn_0"): new_num = 0

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"
        if actual_size == st.session_state.last_pred_size:
            st.session_state.stats["wins"] += 1
            st.session_state.stats["win_streak"] += 1
            st.session_state.stats["loss_streak"] = 0
            status = "✅ WIN"
        else:
            st.session_state.stats["loss"] += 1
            st.session_state.stats["loss_streak"] += 1
            st.session_state.stats["win_streak"] = 0
            status = "❌ LOSS"
        
        st.session_state.history.insert(0, {"#": new_num, "R": status})
        if len(st.session_state.history) > 10: st.session_state.history.pop() # Reduced to 10 for mobile fit
        
        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

    # Reset button (Very small at bottom)
    if st.button("RESET", key="reset_app"):
        st.session_state.clear()
        st.rerun()
