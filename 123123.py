import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

# Compact Page Config for Mobile
st.set_page_config(page_title="91 AI Pro", layout="centered")

# Custom CSS to force single-page view and styling
st.markdown("""
    <style>
    /* Remove extra padding at the top */
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    
    /* Center the big prediction box */
    .pred-box {
        padding: 15px; 
        border-radius: 12px; 
        text-align: center; 
        border: 3px solid white;
        margin-bottom: 10px;
    }
    
    /* Circular Dialer Buttons */
    div.stButton > button {
        border-radius: 50% !important;
        width: 60px !important;
        height: 60px !important;
        font-weight: bold !important;
        font-size: 20px !important;
        color: white !important;
        border: 2px solid white !important;
        margin: 2px auto;
        display: block;
    }
    
    /* Green for 0-4, Red for 5-9 */
    div.stButton > button[key^="btn_0"], div.stButton > button[key^="btn_1"], 
    div.stButton > button[key^="btn_2"], div.stButton > button[key^="btn_3"], 
    div.stButton > button[key^="btn_4"] { background-color: #28a745 !important; }
    
    div.stButton > button[key^="btn_5"], div.stButton > button[key^="btn_6"], 
    div.stButton > button[key^="btn_7"], div.stButton > button[key^="btn_8"], 
    div.stButton > button[key^="btn_9"] { background-color: #dc3545 !important; }
    </style>
""", unsafe_allow_html=True)

# Session States
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
if 'stats' not in st.session_state: st.session_state.stats = {"wins": 0, "loss": 0, "c_win": 0, "c_loss": 0}

# 1. Prediction Result (VISIBLE FIRST)
if 'next_num' in st.session_state:
    color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
    st.markdown(f"""
        <div class="pred-box" style="background-color: {color};">
            <h2 style="color: white; margin: 0;">NEXT: {st.session_state.last_pred_size}</h2>
            <h1 style="color: white; margin: 0; font-size: 50px;">{st.session_state.next_num}</h1>
        </div>
    """, unsafe_allow_html=True)

# 2. Training/Upload (Only shows if no model)
if st.session_state.ai_model is None:
    file = st.file_uploader("Upload Qus.csv", type="csv")
    if file and st.button("🚀 TRAIN"):
        df = pd.read_csv(file)
        if 'content' in df.columns:
            for i in range(1, 6): df[f'p{i}'] = df['content'].shift(i)
            df = df.dropna()
            X, y = df[['p1','p2','p3','p4','p5']], df['content']
            model = GradientBoostingClassifier(n_estimators=100).fit(X, y)
            st.session_state.ai_model = model
            st.rerun()

# 3. Game Interface
elif not st.session_state.last_5:
    init_in = st.text_input("Enter 5 digits to start (e.g. 15152)", max_chars=5)
    if st.button("START"):
        st.session_state.last_5 = [int(d) for d in init_in]
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()
else:
    # 3x3 Dialer Arrangement
    new_num = None
    rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for r in rows:
        cols = st.columns([1, 1, 1, 1])
        for idx, n in enumerate(r):
            if cols[idx].button(str(n), key=f"btn_{n}"): new_num = n
            
    c0_1, c0_2, c0_3, c0_4 = st.columns([1, 1, 1, 1])
    if c0_2.button("0", key="btn_0"): new_num = 0

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"
        if 'last_pred_size' in st.session_state:
            if actual_size == st.session_state.last_pred_size:
                st.session_state.stats["wins"] += 1
                status = "✅ WIN"
            else:
                st.session_state.stats["loss"] += 1
                status = "❌ LOSS"
            st.session_state.history.insert(0, {"Num": new_num, "Res": status})

        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

# 4. Stats & Reset (Small Footer)
if st.session_state.ai_model:
    st.write(f"Wins: {st.session_state.stats['wins']} | Loss: {st.session_state.stats['loss']}")
    if st.button("Reset", key="reset"):
        st.session_state.clear()
        st.rerun()
