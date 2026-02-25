import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

st.set_page_config(page_title="91 AI Ultra-Tracker", layout="wide")

# Initialize Session States
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_5' not in st.session_state:
    st.session_state.last_5 = []
if 'stats' not in st.session_state:
    st.session_state.stats = {"wins": 0, "loss": 0, "c_win": 0, "c_loss": 0, "max_win": 0, "max_loss": 0}

st.title("🚀 91 AI Pro Tracker (Dialer Mode)")

# 1. Training Section
file = st.file_uploader("Upload Qus.csv", type="csv")
if file:
    df = pd.read_csv(file)
    if 'content' in df.columns:
        if st.button("Train Model & Show %"):
            for i in range(1, 6):
                df[f'p{i}'] = df['content'].shift(i)
            df = df.dropna()
            X = df[['p1', 'p2', 'p3', 'p4', 'p5']]
            y = df['content']
            
            model = GradientBoostingClassifier(n_estimators=100)
            model.fit(X, y)
            st.session_state.ai_model = model
            
            # Calculate Accuracy %
            tests = random.sample(range(len(X)), 100)
            score = 0
            for i in tests:
                if model.predict([X.iloc[i]])[0] == y.iloc[i]:
                    score += 1
            st.session_state.accuracy_val = score 
            st.success(f"Model Trained! Accuracy: {st.session_state.accuracy_val}%")

# 2. Prediction Section
if st.session_state.ai_model:
    st.divider()
    
    # Initial 5-digit setup
    if not st.session_state.last_5:
        st.subheader("Setup: Enter first 5 digits (e.g., 15152)")
        init_input = st.text_input("Enter 5 digits then press Start", max_chars=5)
        if st.button("Start Tracking"):
            if len(init_input) == 5 and init_input.isdigit():
                st.session_state.last_5 = [int(d) for d in init_input]
                st.rerun()
            else:
                st.error("Please enter exactly 5 digits.")
    
    # Main Game Loop with Dial Pad Buttons
    else:
        st.write(f"**Current Chain:** `{st.session_state.last_5}`")
        st.subheader("Select New Result:")
        
        # --- Dialer Layout Logic ---
        new_num = None
        
        # Row 1: 1, 2, 3
        r1_col1, r1_col2, r1_col3, r1_empty = st.columns([1, 1, 1, 7])
        if r1_col1.button("1", use_container_width=True, key="btn_1"): new_num = 1
        if r1_col2.button("2", use_container_width=True, key="btn_2"): new_num = 2
        if r1_col3.button("3", use_container_width=True, key="btn_3"): new_num = 3
        
        # Row 2: 4, 5, 6
        r2_col1, r2_col2, r2_col3, r2_empty = st.columns([1, 1, 1, 7])
        if r2_col1.button("4", use_container_width=True, key="btn_4"): new_num = 4
        if r2_col2.button("5", use_container_width=True, key="btn_5"): new_num = 5
        if r2_col3.button("6", use_container_width=True, key="btn_6"): new_num = 6
        
        # Row 3: 7, 8, 9
        r3_col1, r3_col2, r3_col3, r3_empty = st.columns([1, 1, 1, 7])
        if r3_col1.button("7", use_container_width=True, key="btn_7"): new_num = 7
        if r3_col2.button("8", use_container_width=True, key="btn_8"): new_num = 8
        if r3_col3.button("9", use_container_width=True, key="btn_9"): new_num = 9
        
        # Row 4: 0 (Centered)
        r4_empty1, r4_col0, r4_empty2, r4_empty3 = st.columns([1, 1, 1, 7])
        if r4_col0.button("0", use_container_width=True, key="btn_0"): new_num = 0

        # Process choice if any button was pressed
        if new_num is not None:
            # A. Win/Loss Logic
            if 'last_pred_size' in st.session_state:
                actual_size = "SMALL" if new_num <= 4 else "BIG"
                if actual_size == st.session_state.last_pred_size:
                    st.session_state.stats["wins"] += 1
                    st.session_state.stats["c_win"] += 1
                    st.session_state.stats["c_loss"] = 0
                    status = "✅ WIN"
                else:
                    st.session_state.stats["loss"] += 1
                    st.session_state.stats["c_loss"] += 1
                    st.session_state.stats["c_win"] = 0
                    status = "❌ LOSS"
                
                st.session_state.stats["max_win"] = max(st.session_state.stats["max_win"], st.session_state.stats["c_win"])
                st.session_state.stats["max_loss"] = max(st.session_state.stats["max_loss"], st.session_state.stats["c_loss"])
                st.session_state.history.insert(0, {"Number": new_num, "Size": actual_size, "Result": status})

            # B. Update Window (Shift numbers)
            st.session_state.last_5.pop(0)
            st.session_state.last_5.append(new_num)
            
            # C. Generate Next Prediction
            pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
            st.session_state.next_num = pred
            st.session_state.last_pred_size = "SMALL" if pred <= 4 else "BIG"
            st.rerun()

    # 3. COLOR DISPLAY
    if 'next_num' in st.session_state:
        st.divider()
        st.subheader("🔮 NEXT PREDICTION")
        color = "red" if st.session_state.last_pred_size == "BIG" else "green"
        
        st.markdown(f"""
            <div style="background-color: {color}; padding: 30px; border-radius: 15px; text-align: center; border: 5px solid white;">
                <h1 style="color: white; font-size: 60px; margin: 0;">{st.session_state.next_num} - {st.session_state.last_pred_size}</h1>
            </div>
        """, unsafe_allow_html=True)

        # Stats Metrics
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Wins", st.session_state.stats["wins"])
        m2.metric("Current Streak", f"W:{st.session_state.stats['c_win']} / L:{st.session_state.stats['c_loss']}")
        m3.metric("Max Loss Streak", st.session_state.stats["max_loss"])

        # 4. EXCEL DOWNLOAD
        if st.session_state.history:
            st.subheader("📜 History (Last 20)")
            hist_df = pd.DataFrame(st.session_state.history)
            st.table(hist_df)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                hist_df.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 Download History Excel",
                data=output.getvalue(),
                file_name="91_prediction_log.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if st.button("Reset All"):
    st.session_state.clear()
    st.rerun()
