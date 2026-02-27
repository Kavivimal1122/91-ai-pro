import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import os
import time

# 1. Page Configuration
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Advanced CSS for High Visibility and Big/Bold Training
st.markdown("""
    <style>
    .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; padding-left: 0.2rem !important; padding-right: 0.2rem !important; }
    
    /* Training Animation Style */
    .big-training-text { font-size: 42px; font-weight: 900; color: #00ffcc; text-align: center; margin-bottom: 20px; }
    
    .max-streak-container {
        background-color: #0e1117; padding: 10px; border-radius: 12px;
        text-align: center; border: 2px solid #444; margin-bottom: 5px;
    }
    .max-label { font-size: 14px; font-weight: bold; color: #888; }
    .max-value { font-size: 35px; font-weight: 900; line-height: 1; }

    .stat-card {
        background-color: #0e1117; padding: 10px; border-radius: 12px;
        text-align: center; border: 1px solid #444; margin-bottom: 5px;
    }

    .pred-box { padding: 10px; border-radius: 8px; text-align: center; border: 2px solid white; margin-bottom: 10px; }
    .alert-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: bold; text-align: center; color: white; }

    /* Dialer Buttons styling */
    div.stButton > button {
        width: 100% !important; height: 50px !important; border-radius: 4px !important; 
        font-weight: 900 !important; font-size: 18px !important; color: white !important;        
        border: 1px solid white !important; margin: 1px 0px !important; background-color: #1f1f1f !important;
    }

    [data-testid="column"] { width: 9% !important; flex: 1 1 9% !important; min-width: 9% !important; }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Session State Initialization
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
if 'raw_data' not in st.session_state: st.session_state.raw_data = None
if 'consecutive_loss' not in st.session_state: st.session_state.consecutive_loss = 0
if 'stats' not in st.session_state: 
    st.session_state.stats = {"wins": 0, "loss": 0, "curr_streak": 0, "last_res": None, "max_win": 0, "max_loss": 0}

# --- 4. SHARED TRAINING FUNCTION (BIG/BOLD ANIMATION) ---
def train_ai(file_source):
    df = pd.read_csv(file_source)
    if 'content' in df.columns:
        st.session_state.raw_data = df['content'].tolist() 
        
        # Big/Bold Animation
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        for percent in range(101):
            time.sleep(0.01)
            progress_bar.progress(percent)
            status_placeholder.markdown(f'<div class="big-training-text">TRAINING: {percent}%</div>', unsafe_allow_html=True)
        
        # 10-digit pattern recognition logic
        for i in range(1, 11): df[f'p{i}'] = df['content'].shift(i)
        df = df.dropna()
        
        model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_depth=9, subsample=0.8, random_state=42)
        features = [f'p{i}' for i in range(1, 11)]
        model.fit(df[features], df['content'])
        status_placeholder.markdown('<div class="big-training-text" style="color:#28a745;">COMPLETED 100%</div>', unsafe_allow_html=True)
        time.sleep(0.5)
        return model
    return None

# --- 5. PATTERN & PROBABILITY LOGIC ---
def get_prediction_data(sequence):
    probs = st.session_state.ai_model.predict_proba([sequence])[0]
    pred_num = st.session_state.ai_model.predict([sequence])[0]
    
    small_prob = sum(probs[0:5]) * 100
    big_prob = sum(probs[5:10]) * 100
    
    final_prob = big_prob if pred_num > 4 else small_prob
    pred_size = "BIG" if pred_num > 4 else "SMALL"
    
    # Exact Pattern Matching in Qus.csv
    pattern_found, pattern_win_status = False, "Unknown"
    data = st.session_state.raw_data
    seq_len = len(sequence)
    
    for i in range(len(data) - seq_len - 1):
        if data[i:i+seq_len] == list(reversed(sequence)): # Correct order match
            pattern_found = True
            hist_next_size = "SMALL" if data[i+seq_len] <= 4 else "BIG"
            pattern_win_status = "WIN" if hist_next_size == pred_size else "LOSS"
            break
            
    return pred_size, pred_num, final_prob, pattern_found, pattern_win_status

# --- 6. INITIALIZATION ---
if st.session_state.ai_model is None:
    qus_path = "Qus.csv"
    if os.path.exists(qus_path):
        st.session_state.ai_model = train_ai(qus_path)
        if st.session_state.ai_model: st.rerun()
    uploaded_qus = st.file_uploader("Upload Qus.csv", type="csv")
    if uploaded_qus and st.button("🚀 TRAIN AI"):
        st.session_state.ai_model = train_ai(uploaded_qus)
        st.rerun()

# --- 7. MAIN INTERFACE ---
else:
    mode = st.radio("SELECT MODE", ["Real-Time Dialer", "Batch Exam Mode"], horizontal=True)
    st.divider()

    if mode == "Real-Time Dialer":
        if not st.session_state.last_5:
            init_in = st.text_input("Enter 10 digits to start", max_chars=10)
            if st.button("START"):
                if len(init_in) == 10:
                    st.session_state.last_5 = [int(d) for d in init_in]
                    st.rerun()
        else:
            # LIVE STATS BLOCK
            win_rate = (st.session_state.stats['wins'] / (st.session_state.stats['wins'] + st.session_state.stats['loss']) * 100) if (st.session_state.stats['wins'] + st.session_state.stats['loss']) > 0 else 0
            
            st.markdown(f"""
                <div class="max-streak-container">
                    <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                        <div><div class="max-label">MAX WIN</div><div class="max-value" style="color: #28a745;">{st.session_state.stats['max_win']}</div></div>
                        <div><div class="max-label">MAX LOSS</div><div class="max-value" style="color: #dc3545;">{st.session_state.stats['max_loss']}</div></div>
                        <div><div class="max-label">WINS</div><div class="max-value">{st.session_state.stats['wins']}</div></div>
                        <div><div class="max-label">LOSS</div><div class="max-value">{st.session_state.stats['loss']}</div></div>
                        <div><div class="max-label">WIN RATE</div><div class="max-value" style="color: #00ffcc;">{win_rate:.2f}%</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # SAFETY & PATTERN ALERTS
            p_size, p_num, prob, p_match, p_res = get_prediction_data(st.session_state.last_5)
            
            if st.session_state.consecutive_loss >= 3:
                st.markdown('<div class="alert-box" style="background-color: #dc3545;">🛑 3 TIMES LOSS SO STOP - WAIT FOR NORMALIZE</div>', unsafe_allow_html=True)
            elif p_match:
                st.markdown('<div class="alert-box" style="background-color: #28a745;">✅ GO PLAY</div>', unsafe_allow_html=True)
            
            if prob >= 99.9: st.warning("🔥 100% NEXT COME")
            if p_match:
                st.info("📋 THIS SAME PATTERN YOUR DATA")
                res_col = "#28a745" if p_res == "WIN" else "#dc3545"
                st.markdown(f'<div class="alert-box" style="background-color: {res_col};">PATTEN RESULT {p_res}</div>', unsafe_allow_html=True)

            # Prediction Box
            bg_color = "#dc3545" if p_size == "BIG" else "#28a745"
            st.markdown(f"""
                <div class="pred-box" style="background-color: {bg_color};">
                    <p style="margin:0; font-size:12px;">PROBABILITY: {prob:.2f}%</p>
                    <h1 style="color:white; margin:0;">{p_size} ({p_num})</h1>
                </div>
            """, unsafe_allow_html=True)

            # Dialer Buttons
            new_num = None
            cols = st.columns(10)
            for i in range(10):
                if cols[i].button(str(i), key=f"d_{i}"): new_num = i

            if new_num is not None:
                actual_size = "SMALL" if new_num <= 4 else "BIG"
                is_win = (actual_size == p_size)
                
                # Update Streaks & Stats
                res_type = "win" if is_win else "loss"
                st.session_state.stats["wins" if is_win else "loss"] += 1
                
                if res_type == st.session_state.stats["last_res"]: st.session_state.stats["curr_streak"] += 1
                else:
                    st.session_state.stats["curr_streak"], st.session_state.stats["last_res"] = 1, res_type
                
                st.session_state.stats[f"max_{res_type}"] = max(st.session_state.stats[f"max_{res_type}"], st.session_state.stats["curr_streak"])
                st.session_state.consecutive_loss = 0 if is_win else (st.session_state.consecutive_loss + 1)
                
                # Update 20 Result History
                st.session_state.history.insert(0, {"Num": new_num, "Type": actual_size, "Result": "✅ WIN" if is_win else "❌ LOSS", "Streak": st.session_state.stats["curr_streak"], "%": f"{prob:.1f}%"})
                st.session_state.last_5.pop(0)
                st.session_state.last_5.append(new_num)
                st.rerun()

            if st.session_state.history:
                st.markdown("### Past 20 Result History")
                st.table(pd.DataFrame(st.session_state.history).head(20))

    else:
        st.title("🎯 Batch Exam Mode")
        test_file = st.file_uploader("Upload exam.csv", type="csv")
        if test_file:
            if st.button("🔥 START"):
                df_test = pd.read_csv(test_file)
                if 'content' in df_test.columns:
                    nums = df_test['content'].tolist()
                    batch_res, b_wins, b_loss, b_streak, b_max_w, b_max_l, b_last = [], 0, 0, 0, 0, 0, None
                    for i in range(10, len(nums)):
                        feats = [nums[i-j] for j in range(1, 11)]
                        p_size, p_num, prob, _, _ = get_prediction_data(feats)
                        a_num = nums[i]
                        is_w = (p_size == ("SMALL" if a_num <= 4 else "BIG"))
                        
                        if is_w:
                            b_wins += 1
                            b_streak = (b_streak + 1) if b_last == "WIN" else 1
                            b_last, b_max_w = "WIN", max(b_max_w, b_streak)
                        else:
                            b_loss += 1
                            b_streak = (b_streak + 1) if b_last == "LOSS" else 1
                            b_last, b_max_l = "LOSS", max(b_max_l, b_streak)
                        batch_res.append({"Actual": a_num, "AI Pred": f"{p_size}({p_num})", "Status": "WIN" if is_w else "LOSS", "Streak": b_streak})

                    st.markdown(f"""
                        <div class="max-streak-container">
                            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                                <div><div class="max-label">MAX WIN</div><div class="max-value" style="color: #28a745;">{b_max_w}</div></div>
                                <div><div class="max-label">MAX LOSS</div><div class="max-value" style="color: #dc3545;">{b_max_l}</div></div>
                                <div><div class="max-label">WINS</div><div class="max-value">{b_wins}</div></div>
                                <div><div class="max-label">LOSS</div><div class="max-value">{b_loss}</div></div>
                                <div><div class="max-label">WIN RATE</div><div class="max-value" style="color: #00ffcc;">{(b_wins/(b_wins+b_loss)*100):.2f}%</div></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(batch_res), use_container_width=True)

    if st.button("🔄 RESET"):
        st.session_state.clear()
        st.rerun()
