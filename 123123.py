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

# --- 3. FIX: INITIALIZE SESSION STATES (MEMORY) ---
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

# --- 4. PREDICTION & STATS DISPLAY ---
if 'next_num' in st.session_state and st.session_state.ai_model is not None:
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

# --- 5. SETUP WORKFLOW (UPLOAD -> TRAIN -> START) ---
if st.session_state.ai_model is None:
    file = st.file_uploader("Upload Qus.csv", type="csv")
    if file:
        if st.button("🚀 TRAIN HIGH-ACC AI"):
            df = pd.read_csv(file)
            if 'content' in df.columns:
                # Engineering Lags for pattern recognition
                for i in range(1, 6): df[f'p{i}'] = df['content'].shift(i)
                df = df.dropna()
                
                # Advanced Model Setup
                model = GradientBoostingClassifier(
                    n_estimators=1000, 
                    learning_rate=0.02, 
                    max_depth=7, 
                    subsample=0.8, 
                    random_state=42
                )
                model.fit(df[['p1','p2','p3','p4','p5']], df['content'])
                
                st.session_state.accuracy = int(model.score(df[['p1','p2','p3','p4','p5']], df['content']) * 100)
                st.session_state.ai_model = model
                st.success("Training Complete! Now enter starting numbers.")
                st.rerun()

elif not st.session_state.last_5:
    st.info(f"AI Confidence: {st.session_state.accuracy}%")
    init_in = st.text_input("Enter 5 digits (Example: 35125)", max_chars=5)
    if st.button("CONFIRM & START DIALER"):
        if len(init_in) == 5:
            st.session_state.last_5 = [int(d) for d in init_in]
            pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
            st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
            st.rerun()

# --- 6. THE DIALER ---
else:
    new_num = None
    row1 = st.columns(5)
    for i in range(5):
        if row1[i].button(str(i), key=f"btn_{i}"): new_num = i
    
    row2 = st.columns(5)
    for i in range(5, 10):
        if row2[i].button(str(i), key=f"btn_{i}"): new_num = i

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"
        is_win = (actual_size == st.session_state.last_pred_size)
        result_type = "win" if is_win else "loss"
        
        # Streak and Stat Updates
        if is_win: st.session_state.stats["wins"] += 1
        else: st.session_state.stats["loss"] += 1
            
        if result_type == st.session_state.stats["last_result"]:
            st.session_state.stats["current_streak_val"] += 1
        else:
            st.session_state.stats["current_streak_val"] = 1
            st.session_state.stats["last_result"] = result_type

        # Update Maximums
        st.session_state.stats[f"max_{result_type}_streak"] = max(st.session_state.stats[f"max_{result_type}_streak"], st.session_state.stats["current_streak_val"])

        st.session_state.history.insert(0, {"Type": result_type.upper(), "Count": st.session_state.stats["current_streak_val"], "#": new_num})
        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history).head(10))

    if st.button("RESET"):
        st.session_state.clear()
        st.rerun()
