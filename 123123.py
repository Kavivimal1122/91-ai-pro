import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

# 1. Compact Page Config
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Custom CSS
st.markdown("""
    <style>
    .block-container { 
        padding-top: 0.5rem; 
        padding-bottom: 0rem; 
        padding-left: 0.5rem; 
        padding-right: 0.5rem; 
    }
    
    .pred-box {
        padding: 10px; 
        border-radius: 8px; 
        text-align: center; 
        border: 2px solid white;
        margin-bottom: 5px;
    }

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

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Session States
if 'ai_model' not in st.session_state: 
    st.session_state.ai_model = None
if 'history' not in st.session_state: 
    st.session_state.history = []
if 'last_5' not in st.session_state: 
    st.session_state.last_5 = []
if 'stats' not in st.session_state: 
    st.session_state.stats = {"wins": 0, "loss": 0}

# --- 1. PREDICTION ---
if 'next_num' in st.session_state:
    color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
    st.markdown(f"""
        <div class="pred-box" style="background-color: {color};">
            <h2 style="color: white; margin: 0; font-size: 18px;">
                NEXT: {st.session_state.last_pred_size}
            </h2>
            <h1 style="color: white; margin: 0; font-size: 45px;">
                {st.session_state.next_num}
            </h1>
        </div>
    """, unsafe_allow_html=True)

# --- 2. STARTUP LOGIC ---
if st.session_state.ai_model is None:
    file = st.file_uploader("Upload Qus.csv", type="csv")
    if file and st.button("🚀 TRAIN"):
        df = pd.read_csv(file)
        for i in range(1, 6): 
            df[f'p{i}'] = df['content'].shift(i)
        df = df.dropna()
        model = GradientBoostingClassifier(n_estimators=100).fit(
            df[['p1','p2','p3','p4','p5']], df['content']
        )
        st.session_state.ai_model = model
        st.rerun()

elif not st.session_state.last_5:
    init_in = st.text_input("Enter 5 digits (e.g. 35125)", max_chars=5)
    if st.button("CONFIRM"):
        st.session_state.last_5 = [int(d) for d in init_in]
        pred = st.session_state.ai_model.predict(
            [st.session_state.last_5]
        )[0]
        st.session_state.next_num = pred
        st.session_state.last_pred_size = (
            "SMALL" if pred <= 4 else "BIG"
        )
        st.rerun()

# --- 3. THE DIALER ---
else:
    new_num = None

    # Row 1
    col1, col2, col3 = st.columns(3)
    if col1.button("1", key="btn_1"): new_num = 1
    if col2.button("2", key="btn_2"): new_num = 2
    if col3.button("3", key="btn_3"): new_num = 3

    # Row 2
    col1, col2, col3 = st.columns(3)
    if col1.button("4", key="btn_4"): new_num = 4
    if col2.button("5", key="btn_5"): new_num = 5
    if col3.button("6", key="btn_6"): new_num = 6

    # Row 3
    col1, col2, col3 = st.columns(3)
    if col1.button("7", key="btn_7"): new_num = 7
    if col2.button("8", key="btn_8"): new_num = 8
    if col3.button("9", key="btn_9"): new_num = 9

    # Centered 0
    col1, col2, col3 = st.columns(3)
    if col2.button("0", key="btn_0"): new_num = 0

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"

        if actual_size == st.session_state.last_pred_size:
            st.session_state.stats["wins"] += 1
        else:
            st.session_state.stats["loss"] += 1
        
        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)

        pred = st.session_state.ai_model.predict(
            [st.session_state.last_5]
        )[0]

        st.session_state.next_num = pred
        st.session_state.last_pred_size = (
            "SMALL" if pred <= 4 else "BIG"
        )

        st.rerun()

    if st.button("RESET", key="reset_app"):
        st.session_state.clear()
        st.rerun()
