import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

st.set_page_config(page_title="91 AI Ultra-Tracker", layout="wide")

# --- CUSTOM CSS FOR ROUND BUTTONS ---
st.markdown("""
    <style>
    div.stButton > button {
        border-radius: 50% !important;
        width: 60px !important;
        height: 60px !important;
        font-weight: bold !important;
        color: white !important;
        border: 2px solid white !important;
        margin: 5px;
    }
    /* Green for 0-4 */
    div.stButton > button[key^="btn_0"], div.stButton > button[key^="btn_1"], 
    div.stButton > button[key^="btn_2"], div.stButton > button[key^="btn_3"], 
    div.stButton > button[key^="btn_4"] { background-color: #28a745 !important; }
    /* Red for 5-9 */
    div.stButton > button[key^="btn_5"], div.stButton > button[key^="btn_6"], 
    div.stButton > button[key^="btn_7"], div.stButton > button[key^="btn_8"], 
    div.stButton > button[key^="btn_9"] { background-color: #dc3545 !important; }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE ALL SESSION STATES ---
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None
if 'accuracy_val' not in st.session_state:
    st.session_state.accuracy_val = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_5' not in st.session_state:
    st.session_state.last_5 = []
if 'stats' not in st.session_state:
    st.session_state.stats = {"wins": 0, "loss": 0, "c_win": 0, "c_loss": 0, "max_win": 0, "max_loss": 0}

st.title("🚀 91 AI Pro Tracker (Dialer Mode)")

# 1. TRAINING SECTION
file = st.file_uploader("Upload Qus.csv", type="csv")

if file is not None:
    if st.session_state.ai_model is None:
        if st.button("Train Model & Show %"):
            df = pd.read_csv(file)
            if 'content' in df.columns:
                with st.spinner('Training...'):
                    # Data Prep
                    for i in range(1, 6):
                        df[f'p{i}'] = df['content'].shift(i)
                    df = df.dropna()
                    X = df[['p1', 'p2', 'p3', 'p4', 'p5']]
                    y = df['content']
                    
                    # Model
                    model = GradientBoostingClassifier(n_estimators=100)
                    model.fit(X, y)
                    
                    # Accuracy Check
                    tests = random.sample(range(len(X)), 100)
                    score = sum(1 for i in tests if model.predict([X.iloc[i]])[0] == y.iloc[i])
                    
                    # Save to Session
                    st.session_state.ai_model = model
                    st.session_state.accuracy_val = score
                    st.rerun()
            else:
                st.error("CSV must have 'content' header!")

# 2. SHOW ACCURACY IF TRAINED
if st.session_state.accuracy_val is not None:
    st.success(f"Model Trained! Accuracy: {st.session_state.accuracy_val}%")

# 3. MAIN GAME LOGIC
if st.session_state.ai_model is not None:
    st.divider()
    
    # SETUP INITIAL 5 DIGITS
    if not st.session_state.last_5:
        st.subheader("Setup: Enter first 5 digits (e.g., 15152)")
        init_input = st.text_input("Enter 5 digits then press Start", max_chars=5)
        if st.button("Start Tracking"):
            if len(init_input) == 5 and init_input.isdigit():
                st.session_state.last_5 = [int(d) for d in init_input]
                st.rerun()
    
    # DIALER INTERFACE
    else:
        st.write(f"**Current Chain:** `{st.session_state.last_5}`")
        st.subheader("Touch Number for New Result:")
        
        # Grid Layout
        rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        new_num = None
        
        for row in rows:
            cols = st.columns([1, 1, 1, 10])
            for i, num in enumerate(row):
                if cols[i].button(str(num), key=f"btn_{num}"):
                    new_num = num

        cols0 = st.columns([1, 1, 1, 10])
        if cols0[1].button("0", key="btn_0"):
            new_num = 0

        # IF BUTTON CLICKED
        if new_num is not None:
            # Check Prediction Result
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
                
                # Update streaks
                st.session_state.stats["max_win"] = max(st.session_state.stats["max_win"], st.session_state.stats["c_win"])
                st.session_state.stats["max_loss"] = max(st.session_state.stats["max_loss"], st.session_state.stats["c_loss"])
                st.session_state.history.insert(0, {"Number": new_num, "Size": actual_size, "Result": status})

            # Update Chain and Predict
            st.session_state.last_5.pop(0)
            st.session_state.last_5.append(new_num)
            
            pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
            st.session_state.next_num = pred
            st.session_state.last_pred_size = "SMALL" if pred <= 4 else "BIG"
            st.rerun()

    # 4. PREDICTION BOX
    if 'next_num' in st.session_state:
        st.divider()
        color = "red" if st.session_state.last_pred_size == "BIG" else "green"
        st.markdown(f"""
            <div style="background-color: {color}; padding: 25px; border-radius: 15px; text-align: center; border: 4px solid white;">
                <h1 style="color: white; font-size: 50px; margin: 0;">{st.session_state.next_num} - {st.session_state.last_pred_size}</h1>
            </div>
        """, unsafe_allow_html=True)

        # 5. STATS & HISTORY
        if st.session_state.history:
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Wins", st.session_state.stats["wins"])
            m2.metric("Streak", f"W:{st.session_state.stats['c_win']} / L:{st.session_state.stats['c_loss']}")
            m3.metric("Max Loss", st.session_state.stats["max_loss"])
            
            st.subheader("📜 Last 20 Games")
            df_hist = pd.DataFrame(st.session_state.history[:20])
            st.table(df_hist)
            
            # Excel Download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_hist.to_excel(writer, index=False)
            st.download_button("📥 Save History Excel", output.getvalue(), "history.xlsx")

if st.button("Reset All Data"):
    st.session_state.clear()
    st.rerun()
