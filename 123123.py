import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

# 1. Page Config
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Custom CSS for Mobile Single-Page View
st.markdown("""
    <style>
    .block-container { 
        padding-top: 0rem !important; 
        padding-bottom: 0rem !important; 
        padding-left: 0.3rem !important; 
        padding-right: 0.3rem !important; 
    }
    
    .stats-header {
        background-color: #1a1a1a;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 5px;
        border: 1px solid #444;
    }
    .stats-val { font-size: 24px; font-weight: 900; color: white; }

    .pred-box {
        padding: 8px; 
        border-radius: 8px; 
        text-align: center; 
        border: 2px solid white;
        margin-bottom: 5px;
    }

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

    /* Force 5 columns for mobile */
    [data-testid="column"] {
        width: 19% !important;
        flex: 1 1 19% !important;
        min-width: 19% !important;
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Session States
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
if 'stats' not in st.session_state: 
    st.session_state.stats = {"wins": 0, "loss": 0, "current_streak_val": 0, "last_result": None}
if 'accuracy' not in st.session_state: st.session_state.accuracy = 0

# --- 1. OVERALL STATS DISPLAY (TOP) ---
if 'next_num' in st.session_state:
    st.markdown(f"""
        <div class="stats-header">
            <span class="stats-val" style="color: #28a745;">Win={st.session_state.stats['wins']}</span> 
            <span class="stats-val" style="color: white; margin: 0 15px;">|</span>
            <span class="stats-val" style="color: #dc3545;">Loss={st.session_state.stats['loss']}</span>
        </div>
    """, unsafe_allow_html=True)
    
    color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
    st.markdown(f"""
        <div class="pred-box" style="background-color: {color};">
            <p style="color: white; margin: 0; font-size: 14px; font-weight: bold;">NEXT: {st.session_state.last_pred_size}</p>
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
        st.session_state.accuracy = 20 # As seen in your screenshot
        st.session_state.ai_model = model
        st.rerun()
elif not st.session_state.last_5:
    init_in = st.text_input("Enter 5 digits", max_chars=5)
    if st.button("CONFIRM START"):
        st.session_state.last_5 = [int(d) for d in init_in]
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

# --- 3. THE DIALER (0-4 and 5-9) ---
else:
    new_num = None
    c0, c1, c2, c3, c4 = st.columns(5)
    if c0.button("0", key="btn_0"): new_num = 0
    if c1.button("1", key="btn_1"): new_num = 1
    if c2.button("2", key="btn_2"): new_num = 2
    if c3.button("3", key="btn_3"): new_num = 3
    if c4.button("4", key="btn_4"): new_num = 4
    
    c5, c6, c7, c8, c9 = st.columns(5)
    if c5.button("5", key="btn_5"): new_num = 5
    if c6.button("6", key="btn_6"): new_num = 6
    if c7.button("7", key="btn_7"): new_num = 7
    if c8.button("8", key="btn_8"): new_num = 8
    if c9.button("9", key="btn_9"): new_num = 9

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"
        is_win = (actual_size == st.session_state.last_pred_size)
        result_type = "win" if is_win else "loss"
        
        # Win/Loss Counter Logic
        if is_win:
            st.session_state.stats["wins"] += 1
        else:
            st.session_state.stats["loss"] += 1
            
        # Running Streak Counter Logic
        if result_type == st.session_state.stats["last_result"]:
            st.session_state.stats["current_streak_val"] += 1
        else:
            st.session_state.stats["current_streak_val"] = 1
            st.session_state.stats["last_result"] = result_type

        # Add to History with Running Count
        st.session_state.history.insert(0, {
            "Type": result_type.upper(), 
            "Count": st.session_state.stats["current_streak_val"],
            "Number": new_num
        })
        
        # Update Chain & Predict Next
        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

    # --- 4. HISTORY TABLE (STREAK LOGIC) ---
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history).head(15))

    if st.button("RESET", key="reset"):
        st.session_state.clear()
        st.rerun()
