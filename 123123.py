import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import os

# 1. Page Configuration
st.set_page_config(page_title="91 AI Pro - Batch Mode", layout="centered")

# 2. Custom CSS
st.markdown("""
    <style>
    .block-container { padding-top: 1rem !important; }
    .stat-card {
        background-color: #0e1117; padding: 15px; border-radius: 12px;
        text-align: center; border: 2px solid #444; margin-bottom: 10px;
    }
    .stat-label { font-size: 14px; font-weight: bold; color: #888; }
    .stat-value { font-size: 30px; font-weight: 900; }
    .pred-win { color: #28a745; }
    .pred-loss { color: #dc3545; }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Session State Initialization
if 'trained_model' not in st.session_state: st.session_state.trained_model = None

# --- 4. AUTO-TRAINING LOGIC ---
# Automatically looks for Qus.csv to train the model
if st.session_state.trained_model is None:
    st.title("🤖 AI Training")
    # Check if Qus.csv exists in the current folder or via uploader
    qus_file = "Qus.csv"
    
    if os.path.exists(qus_file):
        st.info(f"Found {qus_file}. Training in progress...")
        df_train = pd.read_csv(qus_file)
        
        if 'content' in df_train.columns:
            # Create features: using previous 5 numbers to predict the current one
            for i in range(1, 6): 
                df_train[f'p{i}'] = df_train['content'].shift(i)
            
            df_train = df_train.dropna()
            
            # Training the Gradient Boosting Model
            model = GradientBoostingClassifier(
                n_estimators=500, 
                learning_rate=0.02, 
                max_depth=7, 
                subsample=0.8, 
                random_state=42
            )
            model.fit(df_train[['p1','p2','p3','p4','p5']], df_train['content'])
            
            st.session_state.trained_model = model
            st.success("✅ AI Model Trained Successfully!")
            st.rerun()
        else:
            st.error("Error: Qus.csv must have a 'content' column.")
    else:
        st.warning("Please place 'Qus.csv' in the folder or upload it below to train.")
        uploaded_qus = st.file_uploader("Upload Qus.csv", type="csv")
        if uploaded_qus and st.button("🚀 TRAIN NOW"):
            df_train = pd.read_csv(uploaded_qus)
            # (Same training logic as above)
            for i in range(1, 6): df_train[f'p{i}'] = df_train['content'].shift(i)
            df_train = df_train.dropna()
            model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.02, max_depth=7, subsample=0.8, random_state=42)
            model.fit(df_train[['p1','p2','p3','p4','p5']], df_train['content'])
            st.session_state.trained_model = model
            st.rerun()

# --- 5. BATCH PREDICTION (EXAM MODE) ---
else:
    st.title("🎯 Batch Prediction Mode")
    st.write("Upload your `exam.csv` containing the 500 numbers to get results.")

    test_file = st.file_uploader("Upload Exam CSV", type="csv")
    
    if test_file:
        if st.button("🔥 START BATCH PREDICTION"):
            df_test = pd.read_csv(test_file)
            
            if 'content' in df_test.columns:
                numbers = df_test['content'].tolist()
                results = []
                wins = 0
                losses = 0
                current_streak = 0
                max_win_streak = 0
                max_loss_streak = 0
                last_result = None

                # Process numbers starting from the 6th number (needs 5 previous for prediction)
                for i in range(5, len(numbers)):
                    # Get the 5 previous numbers
                    features = [numbers[i-1], numbers[i-2], numbers[i-3], numbers[i-4], numbers[i-5]]
                    
                    # AI Predicts current number
                    pred_num = st.session_state.trained_model.predict([features])[0]
                    actual_num = numbers[i]
                    
                    pred_size = "SMALL" if pred_num <= 4 else "BIG"
                    actual_size = "SMALL" if actual_num <= 4 else "BIG"
                    
                    is_win = (pred_size == actual_size)
                    status = "WIN" if is_win else "LOSS"
                    
                    # Update Stats
                    if is_win:
                        wins += 1
                        if last_result == "WIN": current_streak += 1
                        else: current_streak = 1
                        last_result = "WIN"
                        max_win_streak = max(max_win_streak, current_streak)
                    else:
                        losses += 1
                        if last_result == "LOSS": current_streak += 1
                        else: current_streak = 1
                        last_result = "LOSS"
                        max_loss_streak = max(max_loss_streak, current_streak)

                    results.append({
                        "Index": i + 1,
                        "Actual Num": actual_num,
                        "AI Predicted": f"{pred_size} ({pred_num})",
                        "Status": status,
                        "Current Streak": current_streak
                    })

                # --- 6. DISPLAY RESULTS ---
                # Top Stats Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">TOTAL WINS</div><div class="stat-value pred-win">{wins}</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">TOTAL LOSS</div><div class="stat-value pred-loss">{losses}</div></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">WIN RATE</div><div class="stat-value" style="color:white;">{(wins/(wins+losses)*100):.1f}%</div></div>', unsafe_allow_html=True)

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">MAX WIN STREAK</div><div class="stat-value pred-win">{max_win_streak}</div></div>', unsafe_allow_html=True)
                with col_s2:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">MAX LOSS STREAK</div><div class="stat-value pred-loss">{max_loss_streak}</div></div>', unsafe_allow_html=True)

                # Results Table
                st.subheader("Detailed Result Log")
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True)

                # Download Results
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 DOWNLOAD FULL RESULT CSV", data=csv, file_name='prediction_results.csv', mime='text/csv')
                
            else:
                st.error("The uploaded file must have a 'content' column.")

    if st.button("🔄 RESET AI MODEL"):
        st.session_state.clear()
        st.rerun()
