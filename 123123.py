import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

st.set_page_config(page_title="91 AI Ultra-Tracker", layout="wide")

# Custom CSS for Round and Colored Buttons
st.markdown("""
    <style>
    div.stButton > button {
        border-radius: 50% !important;
        width: 70px !important;
        height: 70px !important;
        font-weight: bold !important;
        font-size: 20px !important;
        color: white !important;
        border: 2px solid white !important;
    }
    /* Green Buttons 0-4 */
    div.stButton > button[key^="btn_0"], div.stButton > button[key^="btn_1"], 
    div.stButton > button[key^="btn_2"], div.stButton > button[key^="btn_3"], 
    div.stButton > button[key^="btn_4"] {
        background-color: #28a745 !important;
    }
    /* Red Buttons 5-9 */
    div.stButton > button[key^="btn_5"], div.stButton > button[key^="btn_6"], 
    div.stButton > button[key^="btn_7"], div.stButton > button[key^="btn_8"], 
    div.stButton > button[key^="btn_9"] {
        background-color: #dc3545 !important;
    }
    </style>
""", unsafe_allow_html=True)

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
        if st.button("Train Model & Show %", key="train_btn"):
            for i in range(1, 6):
                df[f'p{i}'] = df['content'].shift(i)
            df = df.dropna()
            X = df[['p1', 'p2', 'p3', 'p4', 'p5']]
            y = df['content']
            
            model = GradientBoostingClassifier(n_estimators=100)
            model.fit(X, y)
            st.session_state.ai_model = model
            
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
    
    if not st.session_state.last_5:
        st.subheader("Setup: Enter first 5 digits (e.g., 15152)")
        init_input = st.text_input("Enter 5 digits then press Start", max_chars=5)
        if st.button("Start Tracking", key="start_btn"):
            if len(init_input) == 5 and init_input.isdigit():
                st.session_state.last_5 = [int(d) for d in init_input]
                st.rerun()
    else:
        st.write(f"**Current Chain:** `{st.session_state.last_5}`")
        st.subheader("Select New Result:")
        
        # --- MOBILE DIALER LAYOUT ---
        # Rows: (1,2,3), (4,5,6), (7,8,9), (0)
        btn_rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        new_num = None
        
        # Display 1-9
        for row in btn_rows:
            cols = st.columns([1, 1, 1, 4]) # 3 small columns for buttons
            for idx, num in enumerate(row):
                if cols[idx].button(f"{num}", key=f"btn_{num}"):
                    new_num = num

        # Display 0 in the center of a new row
        cols0 = st.columns([1, 1, 1, 4])
        if cols0[1].button("0", key="btn_0"):
            new_num = 0

        # Process the selection
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

            # B. Update Window
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

        # 4. EXCEL DOWNLOAD & HISTORY
        if st.session_state.history:
            st.subheader("📜 History (Last 20)")
            hist_df = pd.DataFrame(st.session_state.history)
            st.table(hist_df)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                hist_df.to_excel(writer, index=False)
            
            st.download_button("📥 Download History Excel", output.getvalue(), "log.xlsx", key="dl_btn")

if st.button("Reset All", key="reset_btn"):
    st.session_state.clear()
    st.rerun()
