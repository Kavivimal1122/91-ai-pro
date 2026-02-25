import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random

# 1. Page Config for tight mobile display
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Custom CSS for HUGE Maximum Streak Display and Mobile View
st.markdown("""
    <style>
    .block-container { 
        padding-top: 0rem !important; 
        padding-bottom: 0rem !important; 
        padding-left: 0.3rem !important; 
        padding-right: 0.3rem !important; 
    }
    
    .max-streak-container {
        background-color: #0e1117;
        padding: 10px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #444;
        margin-bottom: 5px;
    }
    .max-label { font-size: 12px; font-weight: bold; color: #888; text-transform: uppercase; }
    .max-value { font-size: 40px; font-weight: 900; line-height: 1; }

    .total-stats { font-size: 14px; font-weight: bold; text-align: center; margin-bottom: 5px; color: white; }

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
    st.session_state.stats = {
        "wins": 0, "loss": 0, 
        "current_streak_val": 0, "last_result": None,
        "max_win_streak": 0, "max_loss_streak": 0 
    }
if 'accuracy' not in st.session_state: st.session_state.accuracy = 0

# --- 1. HUGE MAX STREAK DISPLAY (TOP) ---
if 'next_num' in st.session_state:
    st.markdown(f"""
        <div class="max-streak-container">
            <div style="display: flex; justify-content: space-around;">
                <div>
                    <div class="max-label">MAX WIN</div>
                    <div class="max-value" style="color: #28a745;">{st.session_state.stats['max_win_streak']}</div>
                </div>
                <div style="width: 2px; background-color: #444; height: 40px;"></div>
                <div>
                    <div class="max-label">MAX LOSS</div>
                    <div class="max-value" style="color: #dc3545;">{st.session_state.stats['max_loss_streak']}</div>
                </div>
            </div>
        </div>
        <div class="total-stats">
            Win: {st.session_state.stats['wins']} | Loss: {st.session_state.stats['loss']}
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
    if file and st.button("🚀 TRAIN HIGH-ACC AI"):
        df = pd.read_csv(file)
        if 'content' in df.columns:
            # Feature Engineering: Creating Lags for better pattern recognition
            for i in range(1, 6): df[f'p{i}'] = df['content'].shift(i)
            df = df.dropna()
            
            X = df[['p1','p2','p3','p4','p5']]
            y = df['content']
            
            # Advanced Gradient Boosting Setup
            model = GradientBoostingClassifier(
                n_estimators=1000,      # More trees for complex patterns
                learning_rate=0.02,     # Careful learning
                max_depth=7,            # Deeper relationship finding
                subsample=0.8,          # Better generalization
                random_state=42
            )
            model.fit(X, y)
            
            # Simple internal check to show score
            st.session_state.accuracy = int(model.score(X, y) * 100)
            st.session_state.ai_model = model
            st.rerun()

elif not st.session_state.last_5:
    st.info(f"AI Training Confidence: {st.session_state.accuracy}%")
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
    for i in range(5):
        if eval(f"c{i}").button(str(i), key=f"btn_{i}"): new_num = i
    
    c5, c6, c7, c8, c9 = st.columns(5)
    for i in range(5, 10):
        if eval(f"c{i}").button(str(i), key=f"btn_{i}"): new_num = i

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"
        is_win = (actual_size == st.session_state.last_pred_size)
        result_type = "win" if is_win else "loss"
        
        # Stats update
        if is_win: st.session_state.stats["wins"] += 1
        else: st.session_state.stats["loss"] += 1
            
        if result_type == st.session_state.stats["last_result"]:
            st.session_state.stats["current_streak_val"] += 1
        else:
            st.session_state.stats["current_streak_val"] = 1
            st.session_state.stats["last_result"] = result_type

        # Update Maximums
        if result_type == "win":
            st.session_state.stats["max_win_streak"] = max(st.session_state.stats["max_win_streak"], st.session_state.stats["current_streak_val"])
        else:
            st.session_state.stats["max_loss_streak"] = max(st.session_state.stats["max_loss_streak"], st.session_state.stats["current_streak_val"])

        # History and Pred update
        st.session_state.history.insert(0, {"Type": result_type.upper(), "Count": st.session_state.stats["current_streak_val"], "#": new_num})
        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history).head(10))

    if st.button("RESET", key="reset"):
        st.session_state.clear()
        st.rerun()
