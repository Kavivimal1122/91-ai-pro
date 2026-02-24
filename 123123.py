import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random

st.set_page_config(page_title="91 AI Tracker", layout="wide")

# Initialize Session States for Memory
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None
if 'history' not in st.session_state:
    st.session_state.history = []  # Stores last 20 results
if 'last_5' not in st.session_state:
    st.session_state.last_5 = []
if 'stats' not in st.session_state:
    st.session_state.stats = {"wins": 0, "loss": 0, "c_win": 0, "c_loss": 0, "max_win": 0, "max_loss": 0}

st.title("🔥 91 AI Tracker & Auto-Predictor")

# 1. Setup Data
file = st.file_uploader("Upload Qus.csv", type="csv")
if file:
    df = pd.read_csv(file)
    if 'content' in df.columns:
        if st.button("Train Model"):
            for i in range(1, 6):
                df[f'p{i}'] = df['content'].shift(i)
            df = df.dropna()
            X = df[['p1', 'p2', 'p3', 'p4', 'p5']]
            y = df['content']
            model = GradientBoostingClassifier(n_estimators=100)
            model.fit(X, y)
            st.session_state.ai_model = model
            st.success("Model Trained and Ready!")

# 2. Prediction Engine
if st.session_state.ai_model:
    st.divider()
    
    # Input Section
    if not st.session_state.last_5:
        init_input = st.text_input("First Time: Enter last 5 numbers (e.g. 1,1,2,2,5)")
        if st.button("Set Initial Numbers"):
            nums = [int(x.strip()) for x in init_input.split(',')]
            if len(nums) == 5:
                st.session_state.last_5 = nums
                st.rerun()
    else:
        st.write(f"**Current Window:** {st.session_state.last_5}")
        new_num = st.number_input("Enter New Game Result (0-9)", 0, 9, key="new_val")
        
        if st.button("Submit & Predict Next"):
            # A. Check Win/Loss of PREVIOUS prediction
            if 'last_pred_size' in st.session_state:
                actual_size = "SMALL" if new_num <= 4 else "BIG"
                if actual_size == st.session_state.last_pred_size:
                    st.session_state.stats["wins"] += 1
                    st.session_state.stats["c_win"] += 1
                    st.session_state.stats["c_loss"] = 0
                    res_text = "✅ WIN"
                else:
                    st.session_state.stats["loss"] += 1
                    st.session_state.stats["c_loss"] += 1
                    st.session_state.stats["c_win"] = 0
                    res_text = "❌ LOSS"
                
                # Update Max Streaks
                st.session_state.stats["max_win"] = max(st.session_state.stats["max_win"], st.session_state.stats["c_win"])
                st.session_state.stats["max_loss"] = max(st.session_state.stats["max_loss"], st.session_state.stats["c_loss"])
                
                # Update History List
                st.session_state.history.insert(0, {"Result": new_num, "Size": actual_size, "Status": res_text})
                if len(st.session_state.history) > 20:
                    st.session_state.history.pop()

            # B. Update the 5-number window (Remove first, add new)
            st.session_state.last_5.pop(0)
            st.session_state.last_5.append(new_num)
            
            # C. Predict for the NEXT game
            pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
            st.session_state.last_pred_size = "SMALL" if pred <= 4 else "BIG"
            st.session_state.next_pred_num = pred
            st.rerun()

    # 3. Display Results
    if 'next_pred_num' in st.session_state:
        c1, c2, c3 = st.columns(3)
        c1.metric("NEXT PREDICTION", f"{st.session_state.next_pred_num}")
        c2.metric("SIZE", st.session_state.last_pred_size)
        c3.metric("STREAK", f"W:{st.session_state.stats['c_win']} | L:{st.session_state.stats['c_loss']}")

        st.subheader("📊 Streak Records")
        st.write(f"🔥 Max Consecutive Wins: **{st.session_state.stats['max_win']}** | ❄️ Max Consecutive Loss: **{st.session_state.stats['max_loss']}**")

        st.subheader("📜 Last 20 Games History")
        st.table(pd.DataFrame(st.session_state.history))

if st.button("Reset All Data"):
    st.session_state.clear()
    st.rerun()
