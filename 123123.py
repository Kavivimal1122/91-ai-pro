import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random

# 1. Page Configuration
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Custom CSS for Mobile Optimization
st.markdown("""
    <style>
    .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; padding-left: 0.2rem !important; padding-right: 0.2rem !important; }
    
    .max-streak-container {
        background-color: #0e1117; padding: 10px; border-radius: 12px;
        text-align: center; border: 2px solid #444; margin-bottom: 5px;
    }
    .max-label { font-size: 14px; font-weight: bold; color: #888; }
    .max-value { font-size: 45px; font-weight: 900; line-height: 1; }

    .total-stats { font-size: 16px; font-weight: bold; text-align: center; margin-bottom: 5px; color: white; }

    .pred-box { padding: 8px; border-radius: 8px; text-align: center; border: 2px solid white; margin-bottom: 5px; }

    /* Target the text input for keyboard entry */
    input {
        font-size: 24px !important;
        text-align: center !important;
        font-weight: bold !important;
        color: #00ffcc !important;
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Session State Initialization
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
if 'stats' not in st.session_state: 
    st.session_state.stats = {"wins": 0, "loss": 0, "curr_streak": 0, "last_res": None, "max_win": 0, "max_loss": 0}

# --- 4. TOP DISPLAY ---
if st.session_state.ai_model is not None and 'next_num' in st.session_state:
    st.markdown(f"""
        <div class="max-streak-container">
            <div style="display: flex; justify-content: space-around; align-items: center;">
                <div><div class="max-label">MAX WIN</div><div class="max-value" style="color: #28a745;">{st.session_state.stats['max_win']}</div></div>
                <div style="width: 3px; background-color: #444; height: 40px;"></div>
                <div><div class="max-label">MAX LOSS</div><div class="max-value" style="color: #dc3545;">{st.session_state.stats['max_loss']}</div></div>
            </div>
        </div>
        <div class="total-stats">Win: {st.session_state.stats['wins']} | Loss: {st.session_state.stats['loss']}</div>
    """, unsafe_allow_html=True)
    
    color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
    st.markdown(f"""
        <div class="pred-box" style="background-color: {color};">
            <p style="color: white; margin: 0; font-size: 12px; font-weight: bold;">NEXT: {st.session_state.last_pred_size}</p>
            <h1 style="color: white; margin: 0; font-size: 40px;">{st.session_state.next_num}</h1>
        </div>
    """, unsafe_allow_html=True)

# --- 5. WORKFLOW ---
if st.session_state.ai_model is None:
    file = st.file_uploader("Upload Qus.csv", type="csv")
    if file and st.button("🚀 TRAIN AI"):
        df = pd.read_csv(file)
        if 'content' in df.columns:
            for i in range(1, 6): df[f'p{i}'] = df['content'].shift(i)
            df = df.dropna()
            model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.02, max_depth=7, subsample=0.8, random_state=42)
            model.fit(df[['p1','p2','p3','p4','p5']], df['content'])
            st.session_state.ai_model = model
            st.rerun()

elif not st.session_state.last_5:
    init_in = st.text_input("Enter 5 digits", max_chars=5)
    if st.button("START"):
        if len(init_in) == 5:
            st.session_state.last_5 = [int(d) for d in init_in]
            pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
            st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
            st.rerun()

# --- 6. NORMAL KEYBOARD INPUT ---
else:
    # Use text_input so user can use mobile or PC keyboard
    # Entering a number and hitting Enter triggers the logic below
    user_input = st.text_input("Enter Actual Number (0-9):", key="keyboard_input", max_chars=1)

    if user_input:
        if user_input.isdigit():
            new_num = int(user_input)
            actual_size = "SMALL" if new_num <= 4 else "BIG"
            is_win = (actual_size == st.session_state.last_pred_size)
            res_type = "win" if is_win else "loss"
            
            st.session_state.stats["wins" if is_win else "loss"] += 1
            if res_type == st.session_state.stats["last_res"]: 
                st.session_state.stats["curr_streak"] += 1
            else:
                st.session_state.stats["curr_streak"] = 1
                st.session_state.stats["last_res"] = res_type
            
            st.session_state.stats[f"max_{res_type}"] = max(st.session_state.stats[f"max_{res_type}"], st.session_state.stats["curr_streak"])
            st.session_state.history.insert(0, {"Type": res_type.upper(), "Streak": st.session_state.stats["curr_streak"], "Num": new_num})
            
            # Update history and predict next
            st.session_state.last_5.pop(0)
            st.session_state.last_5.append(new_num)
            pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
            st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
            
            # Reset input field by rerunning
            st.rerun()
        else:
            st.error("Please enter a single digit between 0-9")

# --- 7. HISTORY & DOWNLOAD ---
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.table(hist_df.head(10))
    
    csv = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 DOWNLOAD HISTORY", data=csv, file_name='ai_results.csv', mime='text/csv')

if st.button("RESET"):
    st.session_state.clear()
    st.rerun()
