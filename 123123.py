# --- 3. THE DIALER ---
else:
    new_num = None

    # Row 1
    col1, col2, col3 = st.columns(3)
    if col1.button("1", key="btn_1"): new_num = 1
    if col2.button("2", key="btn_2"): new_num = 2
    if col3.button("3", key="btn_3"): new_num = 3

    # Row 2
    col1, col2, col3 = st.columns(3)
    if col1.button("4", key="btn_4"): new_num = 4
    if col2.button("5", key="btn_5"): new_num = 5
    if col3.button("6", key="btn_6"): new_num = 6

    # Row 3
    col1, col2, col3 = st.columns(3)
    if col1.button("7", key="btn_7"): new_num = 7
    if col2.button("8", key="btn_8"): new_num = 8
    if col3.button("9", key="btn_9"): new_num = 9

    # Centered 0 row
    col1, col2, col3 = st.columns(3)
    if col2.button("0", key="btn_0"): new_num = 0

    if new_num is not None:
        actual_size = "SMALL" if new_num <= 4 else "BIG"
        if actual_size == st.session_state.last_pred_size:
            st.session_state.stats["wins"] += 1
        else:
            st.session_state.stats["loss"] += 1
        
        st.session_state.last_5.pop(0)
        st.session_state.last_5.append(new_num)
        pred = st.session_state.ai_model.predict([st.session_state.last_5])[0]
        st.session_state.next_num, st.session_state.last_pred_size = pred, ("SMALL" if pred <= 4 else "BIG")
        st.rerun()

    if st.button("RESET", key="reset_app"):
        st.session_state.clear()
        st.rerun()
