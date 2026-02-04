with st.sidebar:
    st.header("Controls")
    st.caption("Use the sliders to adjust geometry relative to the baseline (dataset mean).")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset sliders"):
            # reset ALL features now
            for c in feature_cols:
                st.session_state[f"off_{c}"] = 0.0
            st.rerun()
    with c2:
        compute = st.button("Compute", type="primary")

    st.divider()

    # --- Key parameters (top 8) ---
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.subheader("Key parameters")
    st.markdown('<div class="small-muted">These are the most influential parameters from the model sensitivity analysis.</div>', unsafe_allow_html=True)

    params = {}

    def slider_for(col: str):
        unit = PARAM_UNITS.get(col, "")
        base_val = float(baseline[col])
        left = float(smin[col] - base_val)
        right = float(smax[col] - base_val)

        key = f"off_{col}"
        if key not in st.session_state:
            st.session_state[key] = 0.0

        offset = st.slider(
            f"{pretty_param_name(col)} ({unit})" if unit else pretty_param_name(col),
            left,
            right,
            float(st.session_state[key]),
            0.001,
            key=key,
        )
        current_val = base_val + offset
        params[col] = current_val
        st.caption(f"Current: **{fmt_with_unit(current_val, unit)}**")

    for col in slider_features:
        slider_for(col)

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("All parameters")
    st.caption("Additional parameters are available for completeness.")

    others = [c for c in feature_cols if c not in slider_features]
    for col in others:
        slider_for(col)

    st.divider()
    st.caption("Built by ebprasad")

