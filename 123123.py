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

st.title("🚀 91 AI Pro Tracker (Color Mode)")

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
            st.session_state.accuracy_val = score # out of 100 is %
            st.success(f"Model Trained! Current Accuracy: {st.session_state.accuracy_val}%")

# 2. Prediction Section
if st.session_state.ai_model:
    st.divider()
    
    if not st.session_state.last_5:
        init_input = st.text_input("Enter first 5 numbers (e.g. 1,1,2,2,5)")
        if st.button("Start Tracking"):
            nums = [int(x.strip()) for x in init_input.split(',')]
            if len(nums) == 5:
                st.session_state.last_5 = nums
                st.rerun()
    else:
        st.write(f"**Current Chain:** `{st.session_state.last_5}`")
        new_num = st.number_input("Enter New Result", 0, 9)
        
        if st.button("Submit & Predict Next"):
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
                
                # Update Records
                st.session_state.stats["max_win"] = max(st.session_state.stats["max_win"], st.session_state.stats["c_win"])
                st.session_state.stats["max_loss"] = max(st.session_state.stats["max_loss"], st.session_state.stats["c_loss"])
                st.session_state.history.insert(0, {"Number": new_num, "Size": actual_size, "Result": status})

            # B. Update Window
            st.session_state.last_5.pop(0)
            st.session_state.last_5.append(new_num)
            
            # C. Next Prediction
            pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
            st.session_state.next_num = pred
            st.session_state.last_pred_size = "SMALL" if pred <= 4 else "BIG"
            st.rerun()

    # 3. COLOR DISPLAY
    if 'next_num' in st.session_state:
        st.subheader("🔮 NEXT PREDICTION")
        color = "red" if st.session_state.last_pred_size == "BIG" else "green"
        
        # Big Red or Small Green Display
        st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h1 style="color: white; margin: 0;">{st.session_state.next_num} - {st.session_state.last_pred_size}</h1>
            </div>
        """, unsafe_allow_html=True)

        # Stats Metrics
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Wins", st.session_state.stats["wins"])
        m2.metric("Current Win Streak", st.session_state.stats["c_win"])
        m3.metric("Max Loss Streak", st.session_state.stats["max_loss"])

        # 4. EXCEL DOWNLOAD
        if st.session_state.history:
            st.subheader("📜 History (Last 20)")
            hist_df = pd.DataFrame(st.session_state.history)
            st.table(hist_df)
            
            # Convert to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                hist_df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            st.download_button(
                label="📥 Download History as Excel",
                data=output.getvalue(),
                file_name="91_prediction_history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if st.button("Reset Everything"):
    st.session_state.clear()
    st.rerun()
