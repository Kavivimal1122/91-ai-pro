import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import random
import io

# PAGE CONFIG
st.set_page_config(page_title="91 AI Pro", layout="centered")

# FULL MOBILE SCREEN CSS (NO SCROLL)
st.markdown("""
<style>

/* Force full mobile height */
html, body, [data-testid="stAppViewContainer"] {
    height: 100vh !important;
    overflow: hidden !important;
}

/* Remove extra spacing */
.block-container {
    padding: 5px 8px 5px 8px !important;
}

/* Compact prediction box */
.pred-box {
    padding: 6px;
    border-radius: 6px;
    text-align: center;
    border: 1px solid white;
    margin-bottom: 4px;
}

/* Compact buttons */
div.stButton > button {
    width: 100% !important;
    height: 52px !important;
    border-radius: 6px !important;
    font-weight: 900 !important;
    font-size: 22px !important;
    color: white !important;
    border: 1px solid white !important;
    margin: 2px 0px !important;
    background-color: #1f1f1f !important;
}

/* Reduce element spacing */
.element-container {
    margin-bottom: 4px !important;
}

/* Smaller history */
[data-testid="stDataFrame"] {
    height: 140px !important;
}

/* Hide Streamlit UI */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# SESSION STATES
if 'ai_model' not in st.session_state: st.session_state.ai_model = None
if 'history' not in st.session_state: st.session_state.history = []
if 'last_5' not in st.session_state: st.session_state.last_5 = []
if 'stats' not in st.session_state:
    st.session_state.stats = {"wins": 0, "loss": 0, "win_streak": 0, "loss_streak": 0}
if 'accuracy' not in st.session_state: st.session_state.accuracy = 0

# PREDICTION
if 'next_num' in st.session_state:
    st.caption(f"🔥 {st.session_state.stats['win_streak']} | ❄️ {st.session_state.stats['loss_streak']}")
    color = "#dc3545" if st.session_state.last_pred_size == "BIG" else "#28a745"
    st.markdown(f"""
        <div class="pred-box" style="background-color:{color};">
            <div style="color:white;font-size:13px;">NEXT: {st.session_state.last_pred_size}</div>
            <div style="color:white;font-size:34px;font-weight:900;">
                {st.session_state.next_num}
            </div>
        </div>
    """, unsafe_allow_html=True)

# STARTUP
if st.session_state.ai_model is None:
    file = st.file_uploader("Upload Qus.csv", type="csv")
    if file and st.button("🚀 TRAIN"):
        df = pd.read_csv(file)
        if 'content' in df.columns:
            for i in range(1, 6):
                df[f'p{i}'] = df['content'].shift(i)
            df = df.dropna()
            model = GradientBoostingClassifier(n_estimators=100).fit(
                df[['p1','p2','p3','p4','p5']], df['content']
            )
            tests = random.sample(range(len(df)), min(100, len(df)))
            score = sum(
                1 for i in tests
                if model.predict([df.iloc[i][['p1','p2','p3','p4','p5']]])[0]
                == df.iloc[i]['content']
            )
            st.session_state.accuracy = score
            st.session_state.ai_model = model
            st.rerun()

elif not st.session_state.last_5:
    st.caption(f"Accuracy: {st.session_state.accuracy}%")
    init_in = st.text_input("Enter 5 digits", max_chars=5)
    if st.button("CONFIRM START"):
        st.session_state.last_5 = [int(d) for d in init_in]
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num = pred
        st.session_state.last_pred_size = ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

# DIALER
else:
    new_num = None

    # Row 1 → 0 1 2 3 4
    r1 = st.columns(5)
    if r1[0].button("0", key="btn_0"): new_num = 0
    if r1[1].button("1", key="btn_1"): new_num = 1
    if r1[2].button("2", key="btn_2"): new_num = 2
    if r1[3].button("3", key="btn_3"): new_num = 3
    if r1[4].button("4", key="btn_4"): new_num = 4

    # Row 2 → 5 6 7 8 9
    r2 = st.columns(5)
    if r2[0].button("5", key="btn_5"): new_num = 5
    if r2[1].button("6", key="btn_6"): new_num = 6
    if r2[2].button("7", key="btn_7"): new_num = 7
    if r2[3].button("8", key="btn_8"): new_num = 8
    if r2[4].button("9", key="btn_9"): new_num = 9

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"

        if actual_size == st.session_state.last_pred_size:
            st.session_state.stats["win_streak"] += 1
            st.session_state.stats["loss_streak"] = 0
            status = "✅ WIN"
        else:
            st.session_state.stats["loss_streak"] += 1
            st.session_state.stats["win_streak"] = 0
            status = "❌ LOSS"

        st.session_state.history.insert(0, {"#": new_num, "Result": status})
        if len(st.session_state.history) > 20:
            st.session_state.history.pop()

        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)

        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num = pred
        st.session_state.last_pred_size = ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

    # Compact History (Fixed Height)
    if st.session_state.history:
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True,
            height=140
        )

    if st.button("RESET ALL", key="reset"):
        st.session_state.clear()
        st.rerun()
