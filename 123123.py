import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random

st.set_page_config(page_title="91 AI Pro", layout="centered")

if 'ready' not in st.session_state:
    st.session_state.ready = False

st.title("🎯 91 AI Pro Predictor")

# Upload Area
file = st.file_uploader("Upload Qus.csv", type="csv")

if file:
    df = pd.read_csv(file)
    
    if 'content' in df.columns:
        # Preparation
        df = df.dropna()
        df['p1'] = df['content'].shift(1)
        df['p2'] = df['content'].shift(2)
        df = df.dropna()
        
        X = df[['p1', 'p2']]
        y = df['content']

        if st.button("Train Model (Check 90% Accuracy)"):
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)
            
            # 50 Question Self-Test
            tests = random.sample(range(len(X)), 50)
            score = 0
            for i in tests:
                if model.predict([X.iloc[i]])[0] == y.iloc[i]:
                    score += 1
            
            acc = (score / 50) * 100
            if acc >= 90:
                st.session_state.ready = True
                st.session_state.ai = model
                st.success(f"Passed! Accuracy: {acc}%")
            else:
                st.error(f"Failed. Accuracy only {acc}%. Add more data to Qus.csv.")
    else:
        st.error("Error: Top row must say 'content'")

# Real Game Button (Unlocks only if passed)
if st.session_state.ready:
    st.divider()
    st.header("🎮 Real Game Prediction")
    val = st.number_input("Enter Last Result (0-9)", 0, 9)
    prev = st.number_input("Enter Previous Result (0-9)", 0, 9)
    
    if st.button("Predict Next"):
        res = st.session_state.ai.predict([[val, prev]])[0]
        size = "SMALL" if res <= 4 else "BIG"
        st.header(f"Next: {res} ({size})")
