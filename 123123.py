import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import os

# 1. Page Configuration
st.set_page_config(page_title="91 AI Pro", layout="centered")

# 2. Custom CSS for Single Row Dialer and Mobile Optimization
st.markdown("""
    <style>
    .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; padding-left: 0.2rem !important; padding-right: 0.2rem !important; }
    
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
    .pred-win { color: #28a745; }
    .pred-loss { color: #dc3545; }

    /* Dialer Buttons styling */
    div.stButton > button {
        width: 100% !important; height: 50px !important; border-radius: 4px !important; 
        font-weight: 900 !important; font-size: 18px !important; color: white !important;        
        border: 1px solid white !important; margin: 1px 0px !important; background-color: #1f1f1f !important;
        padding: 0px !important;
    }

    /* Force 10 columns to stay in one line on mobile */
    [data-testid="column"] { width: 9% !important; flex: 1 1 9% !important; min-width: 9% !important; }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Session State Initialization
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
if 'stats' not in st.session_state: 
    st.session_state.stats = {"wins": 0, "loss": 0, "curr_streak": 0, "last_res": None, "max_win": 0, "max_loss": 0}

# --- 4. SHARED TRAINING FUNCTION (HIGH ACCURACY) ---
def train_ai(file_source):
    df = pd.read_csv(file_source)
    if 'content' in df.columns:
        # 10-digit pattern recognition
        for i in range(1, 11): 
            df[f'p{i}'] = df['content'].shift(i)
        
        df = df.dropna()
        
        # Slower learning rate for precision
        model = GradientBoostingClassifier(
            n_estimators=1000, 
            learning_rate=0.01, 
            max_depth=9, 
            subsample=0.8, 
            random_state=42
        )
        
        features = [f'p{i}' for i in range(1, 11)]
        model.fit(df[features], df['content'])
        return model
    return None

# --- 5. INITIALIZATION ---
if st.session_state.ai_model is None:
    st.title("🤖 AI Initialization")
    qus_path = "Qus.csv"
    if os.path.exists(qus_path):
        with st.spinner("Training 10-Digit Model..."):
            st.session_state.ai_model = train_ai(qus_path)
            if st.session_state.ai_model:
                st.success("✅ Auto-Trained from Qus.csv")
                st.rerun()
    
    uploaded_qus = st.file_uploader("Upload Qus.csv to Start", type="csv")
    if uploaded_qus and st.button("🚀 TRAIN AI"):
        st.session_state.ai_model = train_ai(uploaded_qus)
        st.rerun()

# --- 6. MAIN INTERFACE ---
else:
    mode = st.radio("SELECT MODE", ["Real-Time Dialer", "Batch Exam Mode"], horizontal=True)
    st.divider()

    if mode == "Real-Time Dialer":
        if not st.session_state.last_5:
            init_in = st.text_input("Enter 10 digits to start", max_chars=10)
            if st.button("START"):
                if len(init_in) == 10:
                    st.session_state.last_5 = [int(d) for d in init_in]
                    # Features must be in order: p1 (last), p2, p3... p10
                    pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
                    st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
                    st.rerun()
        else:
            st.markdown(f"""
                <div class="max-streak-container">
                    <div style="display: flex; justify-content: space-around; align-items: center;">
                        <div><div class="max-label">MAX WIN</div><div class="max-value" style="color: #28a745;">{st.session_state.stats['max_win']}</div></div>
                        <div style="width: 3px; background-color: #444; height: 30px;"></div>
                        <div><div class="max-label">MAX LOSS</div><div class="max-value" style="color: #dc3545;">{st.session_state.stats['max_loss']}</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
            st.markdown(f'<div class="pred-box" style="background-color: {color};"><h1 style="color:white; margin:0;">{st.session_state.last_pred_size} ({st.session_state.next_num})</h1></div>', unsafe_allow_html=True)

            new_num = None
            cols = st.columns(10)
            for i in range(10):
                if cols[i].button(str(i), key=f"d_{i}"): new_num = i

            if new_num is not None:
                actual_size = "SMALL" if new_num <= 4 else "BIG"
                is_win = (actual_size == st.session_state.last_pred_size)
                res_type = "win" if is_win else "loss"
                st.session_state.stats["wins" if is_win else "loss"] += 1
                if res_type == st.session_state.stats["last_res"]: st.session_state.stats["curr_streak"] += 1
                else:
                    st.session_state.stats["curr_streak"] = 1
                    st.session_state.stats["last_res"] = res_type
                st.session_state.stats[f"max_{res_type}"] = max(st.session_state.stats[f"max_{res_type}"], st.session_state.stats["curr_streak"])
                st.session_state.history.insert(0, {"Type": res_type.upper(), "Streak": st.session_state.stats["curr_streak"], "Num": new_num})
                
                st.session_state.last_5.pop(0)
                st.session_state.last_5.append(new_num)
                pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
                st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
                st.rerun()

            if st.session_state.history: st.table(pd.DataFrame(st.session_state.history).head(10))

    else:
        st.title("🎯 Batch Exam Mode")
        test_file = st.file_uploader("Upload exam.csv", type="csv")
        if test_file:
            if st.button("🔥 START"):
                df_test = pd.read_csv(test_file)
                if 'content' in df_test.columns:
                    nums = df_test['content'].tolist()
                    batch_res, b_wins, b_loss, b_curr, b_max_w, b_max_l, b_last = [], 0, 0, 0, 0, 0, None
                    for i in range(10, len(nums)):
                        feats = [nums[i-j] for j in range(1, 11)]
                        p_num = st.session_state.ai_model.predict([feats])[0]
                        a_num, p_sz = nums[i], ("SMALL" if p_num <= 4 else "BIG")
                        a_sz = ("SMALL" if a_num <= 4 else "BIG")
                        is_w = (p_sz == a_sz)
                        status = "WIN" if is_w else "LOSS"
                        if is_w:
                            b_wins += 1
                            b_curr = (b_curr + 1) if b_last == "WIN" else 1
                            b_last, b_max_w = "WIN", max(b_max_w, b_curr)
                        else:
                            b_loss += 1
                            b_curr = (b_curr + 1) if b_last == "LOSS" else 1
                            b_last, b_max_l = "LOSS", max(b_max_l, b_curr)
                        batch_res.append({"Actual": a_num, "AI Pred": f"{p_sz}({p_num})", "Status": status, "Streak": b_curr})

                    st.markdown(f"""
                        <div class="max-streak-container">
                            <div style="display: flex; justify-content: space-around; align-items: center;">
                                <div><div class="max-label">MAX WIN</div><div class="max-value" style="color: #28a745;">{b_max_w}</div></div>
                                <div style="width: 3px; background-color: #444; height: 30px;"></div>
                                <div><div class="max-label">MAX LOSS</div><div class="max-value" style="color: #dc3545;">{b_max_l}</div></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(batch_res), use_container_width=True)
                else: st.error("CSV must have 'content' column.")

    if st.button("🔄 RESET"):
        st.session_state.clear()
        st.rerun()
