import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import os

# 1. Page Configuration
st.set_page_config(page_title="91 AI Pro - Final", layout="centered")

# 2. Custom CSS for Visual Clarity
st.markdown("""
    <style>
    .block-container { padding-top: 0.5rem !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    .stat-card {
        background-color: #111; padding: 15px; border-radius: 10px;
        text-align: center; border: 1px solid #333; margin-bottom: 10px;
    }
    .stat-label { font-size: 12px; font-weight: bold; color: #888; text-transform: uppercase; }
    .stat-value { font-size: 28px; font-weight: 900; }
    .win-text { color: #28a745; }
    .loss-text { color: #dc3545; }
    .pred-box { padding: 12px; border-radius: 8px; text-align: center; border: 2px solid white; margin-bottom: 10px; }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Session State
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
if 'stats' not in st.session_state: 
    st.session_state.stats = {"wins": 0, "loss": 0, "curr_streak": 0, "last_res": None, "max_win": 0, "max_loss": 0}

# --- 4. CORE ALGORITHM (No Changes) ---
def train_logic(file):
    df = pd.read_csv(file)
    if 'content' in df.columns:
        for i in range(1, 6): df[f'p{i}'] = df['content'].shift(i)
        df = df.dropna()
        model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.02, max_depth=7, subsample=0.8, random_state=42)
        model.fit(df[['p1','p2','p3','p4','p5']], df['content'])
        return model
    return None

# --- 5. INITIALIZATION ---
if st.session_state.ai_model is None:
    st.header("🤖 AI Training Center")
    qus_file = "Qus.csv"
    if os.path.exists(qus_file):
        with st.spinner("Processing Qus.csv..."):
            st.session_state.ai_model = train_logic(qus_file)
            if st.session_state.ai_model: st.rerun()
    
    up_qus = st.file_uploader("Upload Qus.csv to Train", type="csv")
    if up_qus and st.button("🚀 TRAIN MODEL"):
        st.session_state.ai_model = train_logic(up_qus)
        st.rerun()

# --- 6. MAIN APP INTERFACE ---
else:
    mode = st.selectbox("CHOOSE OPERATION", ["Batch Analysis (exam.csv)", "Live Manual Play"])
    st.divider()

    if mode == "Batch Analysis (exam.csv)":
        st.subheader("📁 500 Rounds Batch Processor")
        exam_file = st.file_uploader("Upload exam.csv", type="csv")
        
        if exam_file:
            if st.button("📊 GENERATE RESULTS"):
                df_exam = pd.read_csv(exam_file)
                if 'content' in df_exam.columns:
                    nums = df_exam['content'].tolist()
                    batch_data, b_wins, b_loss, b_streak, b_max_w, b_max_l, b_last = [], 0, 0, 0, 0, 0, None

                    for i in range(5, len(nums)):
                        feats = [nums[i-1], nums[i-2], nums[i-3], nums[i-4], nums[i-5]]
                        p_val = st.session_state.ai_model.predict([feats])[0]
                        a_val = nums[i]
                        p_sz, a_sz = ("SMALL" if p_val <= 4 else "BIG"), ("SMALL" if a_val <= 4 else "BIG")
                        is_w = (p_sz == a_sz)
                        
                        if is_w:
                            b_wins += 1
                            b_streak = (b_streak + 1) if b_last == "WIN" else 1
                            b_last, b_max_w = "WIN", max(b_max_w, b_streak)
                        else:
                            b_loss += 1
                            b_streak = (b_streak + 1) if b_last == "LOSS" else 1
                            b_last, b_max_l = "LOSS", max(b_max_l, b_streak)

                        batch_data.append({"No": i+1, "Actual": a_val, "AI Prediction": f"{p_sz}({p_val})", "Result": "✅ WIN" if is_w else "❌ LOSS", "Streak": b_streak})

                    # Display Visual Statistics
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f'<div class="stat-card"><div class="stat-label">Wins</div><div class="stat-value win-text">{b_wins}</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="stat-card"><div class="stat-label">Loss</div><div class="stat-value loss-text">{b_loss}</div></div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="stat-card"><div class="stat-label">Rate</div><div class="stat-value">{(b_wins/(b_wins+b_loss)*100):.1f}%</div></div>', unsafe_allow_html=True)

                    st.markdown("---")
                    st.dataframe(pd.DataFrame(batch_data), use_container_width=True)
                    
                    # Reference Image for Batch Success
                    
                    
                    st.download_button("📥 DOWNLOAD LOG", pd.DataFrame(batch_data).to_csv(index=False).encode('utf-8'), "exam_output.csv", "text/csv")
                else:
                    st.error("Error: CSV must have 'content' column.")

    else:
        # Live Play Code (Original Dialer Function)
        if not st.session_state.last_5:
            init = st.text_input("Enter 5 numbers to start", max_chars=5)
            if st.button("START"):
                if len(init) == 5:
                    st.session_state.last_5 = [int(d) for d in init]
                    pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
                    st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
                    st.rerun()
        else:
            color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
            st.markdown(f'<div class="pred-box" style="background-color: {color};"><h1>{st.session_state.last_pred_size} ({st.session_state.next_num})</h1></div>', unsafe_allow_html=True)
            
            new_val = None
            cols = st.columns(10)
            for i in range(10):
                if cols[i].button(str(i), key=f"d_{i}"): new_val = i

            if new_val is not None:
                is_w = (("SMALL" if new_val <= 4 else "BIG") == st.session_state.last_pred_size)
                st.session_state.stats["wins" if is_w else "loss"] += 1
                st.session_state.history.insert(0, {"Num": new_val, "Status": "WIN" if is_w else "LOSS"})
                st.session_state.last_5.pop(0)
                st.session_state.last_5.append(new_val)
                pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
                st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
                st.rerun()
            
            if st.session_state.history:
                st.table(pd.DataFrame(st.session_state.history).head(5))

    if st.button("🔄 FULL RESET"):
        st.session_state.clear()
        st.rerun()
