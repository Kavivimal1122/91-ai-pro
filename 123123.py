import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# 1. Page Configuration
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Custom CSS for Joined Excel Grid
st.markdown("""
    <style>
    /* Main container padding */
    .block-container { padding: 0.5rem !important; }

    /* Pred-box styling */
    .pred-box { 
        padding: 10px; 
        border-radius: 8px; 
        text-align: center; 
        border: 2px solid white; 
        margin-bottom: 20px; 
    }

    /* EXCEL GRID BUTTONS: Joint styling */
    div.stButton > button {
        width: 100% !important; 
        height: 60px !important; 
        border-radius: 0px !important; /* Square corners for joining */
        font-weight: 900 !important; 
        font-size: 24px !important; 
        color: black !important;        
        background-color: #ffff00 !important; /* Yellow background */
        border: 1px solid black !important; /* Grid lines */
        margin: 0px !important;
        padding: 0px !important;
    }

    /* Remove the default gap between Streamlit columns to join buttons */
    [data-testid="column"] {
        padding: 0px !important;
        margin: 0px !important;
    }
    
    [data-testid="stHorizontalBlock"] {
        gap: 0px !important;
    }

    /* UI Clean up */
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
    color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
    st.markdown(f"""
        <div class="pred-box" style="background-color: {color};">
            <p style="color: white; margin: 0; font-size: 14px; font-weight: bold;">AI PREDICTION</p>
            <h1 style="color: white; margin: 0; font-size: 45px;">{st.session_state.last_pred_size} ({st.session_state.next_num})</h1>
        </div>
    """, unsafe_allow_html=True)

# --- 5. WORKFLOW ---
if st.session_state.ai_model is None:
    file = st.file_uploader("Upload Game Data (CSV)", type="csv")
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
    init_in = st.text_input("Enter last 5 numbers", max_chars=5)
    if st.button("START"):
        if len(init_in) == 5:
            st.session_state.last_5 = [int(d) for d in init_in]
            pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
            st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
            st.rerun()

# --- 6. JOINED EXCEL GRID (0-9) ---
else:
    new_num = None
    
    # Row 1: Numbers 0 to 4
    row1 = st.columns(5)
    for i in range(5):
        if row1[i].button(str(i), key=f"num_{i}"):
            new_num = i
            
    # Row 2: Numbers 5 to 9
    row2 = st.columns(5)
    for i in range(5, 10):
        if row2[i-5].button(str(i), key=f"num_{i}"):
            new_num = i

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"
        is_win = (actual_size == st.session_state.last_pred_size)
        res_type = "win" if is_win else "loss"
        
        st.session_state.stats["wins" if is_win else "loss"] += 1
        st.session_state.history.insert(0, {"Result": actual_size, "Num": new_num, "Status": res_type.upper()})
        
        # Update AI
        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

    # --- 7. HISTORY TABLE ---
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history).head(5))

    if st.button("RESET"):
        st.session_state.clear()
        st.rerun()
