import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random

st.set_page_config(page_title="91 AI Ultra", layout="centered")

# This keeps the model active after training
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None

st.title("🔥 91 AI Ultra-Predictor")

file = st.file_uploader("Upload Qus.csv", type="csv")

if file:
    df = pd.read_csv(file)
    if 'content' in df.columns:
        # Deep Memory: Look at the last 5 numbers
        for i in range(1, 6):
            df[f'p{i}'] = df['content'].shift(i)
        df = df.dropna()
        
        X = df[['p1', 'p2', 'p3', 'p4', 'p5']]
        y = df['content']

        if st.button("Deep Train Now"):
            model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
            model.fit(X, y)
            
            # Show the accuracy but don't hide the game
            tests = random.sample(range(len(X)), 50)
            score = 0
            for i in tests:
                if model.predict([X.iloc[i]])[0] == y.iloc[i]:
                    score += 1
            
            st.session_state.ai_model = model
            st.success(f"Model Ready! Accuracy: {(score / 50) * 100}%")
    else:
        st.error("Header must be 'content'")

# THE FIX: This box will now appear as soon as the model is trained!
if st.session_state.ai_model is not None:
    st.divider()
    st.header("🎮 Get Next Prediction")
    st.write("Type the last 5 game results below:")
    
    user_input = st.text_input("Example: 1, 5, 0, 2, 9", "")
    
    if st.button("Predict Next Result"):
        try:
            # Turn your typing into a list of numbers
            val_list = [int(x.strip()) for x in user_input.split(',')]
            
            if len(val_list) == 5:
                prediction = st.session_state.ai_model.predict([val_list])[0]
                size = "SMALL (0-4)" if prediction <= 4 else "BIG (5-9)"
                
                st.subheader(f"🎯 Next Number: {prediction}")
                st.header(f"✨ Result: {size}")
            else:
                st.warning("Please enter exactly 5 numbers.")
        except:
            st.error("Use commas between numbers (Example: 1,2,3,4,5)")
