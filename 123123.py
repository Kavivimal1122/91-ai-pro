import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random

st.set_page_config(page_title="91 AI Ultra", layout="centered")

if 'ready' not in st.session_state:
    st.session_state.ready = False

st.title("🔥 91 AI Ultra-Predictor (90% Goal)")

file = st.file_uploader("Upload Qus.csv", type="csv")

if file:
    df = pd.read_csv(file)
    if 'content' in df.columns:
        # 1. Deep Memory: Look at the last 5 numbers instead of 2
        for i in range(1, 6):
            df[f'p{i}'] = df['content'].shift(i)
        df = df.dropna()
        
        # 2. Features and Target
        X = df[['p1', 'p2', 'p3', 'p4', 'p5']]
        y = df['content']

        if st.button("Deep Train (Target 90%)"):
            # Using a stronger model (Gradient Boosting) for complex patterns
            model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
            model.fit(X, y)
            
            # 50 Question Self-Test
            tests = random.sample(range(len(X)), 50)
            score = 0
            for i in tests:
                # Predicting the exact number (0-9)
                if model.predict([X.iloc[i]])[0] == y.iloc[i]:
                    score += 1
            
            acc = (score / 50) * 100
            st.session_state.ai = model
            
            if acc >= 90:
                st.session_state.ready = True
                st.success(f"🎯 EXCELLENT! Accuracy: {acc}%")
            else:
                st.warning(f"Accuracy: {acc}%. To increase, upload more rows to Qus.csv (Target: 15,000+ rows).")
    else:
        st.error("Header must be 'content'")

if st.session_state.ready:
    st.divider()
    st.header("🎮 Real Game Prediction")
    inputs = st.text_input("Enter last 5 numbers (comma separated, e.g. 1,5,0,2,9)")
    
    if st.button("Predict Next"):
        try:
            val_list = [int(x.strip()) for x in inputs.split(',')]
            res = st.session_state.ai.predict([val_list])[0]
            size = "SMALL" if res <= 4 else "BIG"
            st.header(f"Result: {res} ({size})")
        except:
            st.error("Please enter exactly 5 numbers separated by commas.")
