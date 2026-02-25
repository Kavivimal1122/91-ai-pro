import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

# 1. Page Configuration
st.set_page_config(page_title="91 AI Pro Tracker", layout="wide")

# 2. Custom CSS for Mobile Dialer Styling
st.markdown("""
    <style>
    div.stButton > button {
        border-radius: 50% !important;
        width: 65px !important;
        height: 65px !important;
        font-weight: bold !important;
        font-size: 22px !important;
        color: white !important;
        border: 2px solid white !important;
        margin: 5px;
    }
    /* Green buttons 0-4 */
    div.stButton > button[key^="btn_0"], div.stButton > button[key^="btn_1"], 
    div.stButton > button[key^="btn_2"], div.stButton > button[key^="btn_3"], 
    div.stButton > button[key^="btn_4"] { background-color: #28a745 !important; }
    
    /* Red buttons 5-9 */
    div.stButton > button[key^="btn_5"], div.stButton > button[key^="btn_6"], 
    div.stButton > button[key^="btn_7"], div.stButton > button[key^="btn_8"], 
    div.stButton > button[key^="btn_9"] { background-color: #dc3545 !important; }
    </style>
""", unsafe_allow_html=True)

# 3. Initialize Session States (The App's Memory)
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

st.title("🚀 91 AI Pro Tracker")

# 4. Data Upload & Training Logic
if st.session_state.ai_model is None:
    file = st.file_uploader("Upload Qus.csv", type="csv")
    if file is not None:
        if st.button("🚀 START TRAINING NOW"):
            df = pd.read_csv(file)
            if 'content' in df.columns:
                with st.spinner('AI is reading your data 100 times...'):
                    # Pattern Preparation
                    for i in range(1, 6):
                        df[f'p{i}'] = df['content'].shift(i)
                    df = df.dropna()
                    X = df[['p1', 'p2', 'p3', 'p4', 'p5']]
                    y = df['content']
                    
                    # Gradient Boosting Model
                    model = GradientBoostingClassifier(n_estimators=100)
                    model.fit(X, y)
                    
                    # Test Accuracy
                    tests = random.sample(range(len(X)), 100)
                    score = sum(1 for i in tests if model.predict([X.iloc[i]])[0] == y.iloc[i])
                    
                    # Save to Memory
                    st.session_state.ai_model = model
                    st.session_state.accuracy_val = score
                    st.rerun()
            else:
                st.error("Missing column: 'content'")
else:
    # Show status once trained
    st.success(f"✅ AI Model Active | Accuracy: {st.session_state.accuracy_val}%")
    if st.button("🗑️ Reset & Upload New Data"):
        st.session_state.clear()
        st.rerun()

# 5. Main Game Interface
if st.session_state.ai_model is not None:
    st.divider()
    
    # Step 1: Initial 5 Numbers
    if not st.session_state.last_5:
        st.subheader("Setup: Enter last 5 results (e.g. 15152)")
        init_input = st.text_input("Enter 5 digits", max_chars=5)
        if st.button("Confirm & Start"):
            if len(init_input) == 5 and init_input.isdigit():
                st.session_state.last_5 = [int(d) for d in init_input]
                st.rerun()
    
    # Step 2: The Dialer
    else:
        st.write(f"**Sequence:** `{st.session_state.last_5}`")
        st.subheader("Touch Result:")
        
        # Grid arrangement
        rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        new_num = None
        
        for row in rows:
            cols = st.columns([1, 1, 1, 10])
            for idx, num in enumerate(row):
                if cols[idx].button(str(num), key=f"btn_{num}"):
                    new_num = num

        cols0 = st.columns([1, 1, 1, 10])
        if cols0[1].button("0", key="btn_0"):
            new_num = 0

        # When a number is touched
        if new_num is not None:
            # Win/Loss Tracker
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

            # Shift the window and predict next
            st.session_state.last_5.pop(0)
            st.session_state.last_5.append(new_num)
            
            pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
            st.session_state.next_num = pred
            st.session_state.last_pred_size = "SMALL" if pred <= 4 else "BIG"
            st.rerun()

    # 6. Display Next Prediction
    if 'next_num' in st.session_state:
        st.divider()
        st.subheader("🔮 NEXT PREDICTION")
        color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
        
        st.markdown(f"""
            <div style="background-color: {color}; padding: 30px; border-radius: 20px; text-align: center; border: 5px solid white;">
                <h1 style="color: white; font-size: 60px; margin: 0;">{st.session_state.next_num} - {st.session_state.last_pred_size}</h1>
            </div>
        """, unsafe_allow_html=True)

        # 7. Stats and Table
        if st.session_state.history:
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Wins", st.session_state.stats["wins"])
            c2.metric("Loss", st.session_state.stats["loss"])
            c3.metric("Max Loss", st.session_state.stats["max_loss"])
            
            st.subheader("📜 Last 20 Results")
            df_hist = pd.DataFrame(st.session_state.history[:20])
            st.table(df_hist)
            
            # Excel Generation
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_hist.to_excel(writer, index=False)
            st.download_button("📥 Save to Excel", output.getvalue(), "91_log.xlsx")
