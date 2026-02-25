import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

# 1. Page Config for Mobile
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Custom CSS for Solid Block Buttons and Large White Numbers
st.markdown("""
    <style>
    /* Prevent scrolling and center content */
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    
    /* Result box at the top */
    .pred-box {
        padding: 15px; 
        border-radius: 12px; 
        text-align: center; 
        border: 4px solid white;
        margin-bottom: 15px;
    }

    /* Large Block Buttons with Big White Numbers */
    div.stButton > button {
        width: 100% !important;
        height: 80px !important;
        border-radius: 10px !important; 
        font-weight: 900 !important;   
        font-size: 35px !important;   
        color: white !important;       
        border: 2px solid white !important;
        margin-bottom: 10px;
    }
    
    /* Green Block Buttons (0-4) */
    div.stButton > button[key^="btn_0"], div.stButton > button[key^="btn_1"], 
    div.stButton > button[key^="btn_2"], div.stButton > button[key^="btn_3"], 
    div.stButton > button[key^="btn_4"] { 
        background-color: #28a745 !important; 
    }
    
    /* Red Block Buttons (5-9) */
    div.stButton > button[key^="btn_5"], div.stButton > button[key^="btn_6"], 
    div.stButton > button[key^="btn_7"], div.stButton > button[key^="btn_8"], 
    div.stButton > button[key^="btn_9"] { 
        background-color: #dc3545 !important; 
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
if 'stats' not in st.session_state: st.session_state.stats = {"wins": 0, "loss": 0}

# --- 1. RESULT DISPLAY (TOP) ---
if 'next_num' in st.session_state:
    color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
    st.markdown(f"""
        <div class="pred-box" style="background-color: {color};">
            <h2 style="color: white; margin: 0; font-size: 22px;">NEXT: {st.session_state.last_pred_size}</h2>
            <h1 style="color: white; margin: 0; font-size: 60px;">{st.session_state.next_num}</h1>
        </div>
    """, unsafe_allow_html=True)

# --- 2. SETUP & TRAINING ---
if st.session_state.ai_model is None:
    file = st.file_uploader("Upload Qus.csv", type="csv")
    if file and st.button("🚀 TRAIN MODEL"):
        df = pd.read_csv(file)
        if 'content' in df.columns:
            for i in range(1, 6): df[f'p{i}'] = df['content'].shift(i)
            df = df.dropna()
            X, y = df[['p1','p2','p3','p4','p5']], df['content']
            model = GradientBoostingClassifier(n_estimators=100).fit(X, y)
            st.session_state.ai_model = model
            st.rerun()
elif not st.session_state.last_5:
    init_in = st.text_input("Enter 5 digits (e.g. 35125)", max_chars=5)
    if st.button("START TRACKING"):
        if len(init_in) == 5:
            st.session_state.last_5 = [int(d) for d in init_in]
            pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
            st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
            st.rerun()

# --- 3. DIALER BUTTONS (CENTER) ---
else:
    new_num = None
    # Grid layout: 1-2-3, 4-5-6, 7-8-9, 0
    rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for r in rows:
        cols = st.columns(3)
        for idx, n in enumerate(r):
            if cols[idx].button(str(n), key=f"btn_{n}"): new_num = n
            
    c_empty1, c0, c_empty2 = st.columns(3)
    if c0.button("0", key="btn_0"): new_num = 0

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"
        if actual_size == st.session_state.last_pred_size:
            st.session_state.stats["wins"] += 1
        else:
            st.session_state.stats["loss"] += 1
        
        # Update logic
        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

# --- 4. FOOTER ---
if st.session_state.ai_model:
    st.write(f"W: {st.session_state.stats['wins']} | L: {st.session_state.stats['loss']}")
    if st.button("RESET"):
        st.session_state.clear()
        st.rerun()
