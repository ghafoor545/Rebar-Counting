import os
import time
import base64

import cv2
import numpy as np
import streamlit as st

from config import PER_PAGE
from helpers import do_rerun, scroll_top, show_image_full_width, flash, show_flash
from utils import fmt_local_time
from auth import (
    create_user,
    get_user_by_login,
    verify_password,
    save_persistent_session,
    clear_persistent_session,
)
from detector import (
    model,
    fetch_snapshot,
    detect_rebars,
    record_detection,
    list_detections,
    delete_detection,
    img_to_data_uri,
    file_to_data_uri,
)
from oak_utils import (
    ensure_oak_device,
    get_oak_frame,
    grab_oak_frame,
    close_oak_device,
)


def page_login():
    ss = st.session_state
    scroll_top()

    st.markdown(
        """
<style>
div.block-container{
  min-height: 100vh !important;
  display: grid !important;
  place-items: center !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    show_flash()

    with st.form("login_form", clear_on_submit=False):
        st.markdown(
            """
            <div style="display:grid;place-items:center;margin-bottom:6px;filter:drop-shadow(0 0 16px rgba(34,211,238,0.65));">
              <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                <defs><linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stop-color="#22d3ee"/><stop offset="50%" stop-color="#8b5cf6"/><stop offset="100%" stop-color="#34d399"/>
                </linearGradient></defs>
                <path d="M12 20h10l4-6h12l4 6h10a6 6 0 0 1 6 6v18a6 6 0 0 1 6 6H12a6 6 0 0 1-6-6V26a6 6 0 0 1 6-6z" stroke="url(#g1)" stroke-width="3" />
                <circle cx="32" cy="35" r="10" stroke="url(#g1)" stroke-width="3"/>
                <circle cx="32" cy="35" r="4" fill="#20e3b2"/>
              </svg>
            </div>
            <h2 class="gradient-heading" style="text-align:center;margin:0;">Rebar Counting</h2>
            <p style="text-align:center;color:#dbe6ff;margin:6px 0 14px;">Secure login to access your dashboard.</p>
            """,
            unsafe_allow_html=True,
        )

        identifier = st.text_input(
            "Username / Email", key="lg_user", placeholder="Username / Email"
        )
        password = st.text_input(
            "Password", key="lg_pass", type="password", placeholder="Password"
        )

        col_l, col_r = st.columns([1, 1])
        with col_l:
            remember = st.checkbox("Remember Me", value=True, key="lg_remember")
        with col_r:
            st.markdown(
                '<div style="text-align:right;"><a style="color:#a8c7ff;text-decoration:none;font-weight:800;" href="#">Forgot <b>Password?</b></a></div>',
                unsafe_allow_html=True,
            )

        b1, b2 = st.columns([2, 1])
        with b1:
            submit_login = st.form_submit_button("Sign In", width="stretch")
        with b2:
            goto_signup = st.form_submit_button("Sign Up", width="stretch")

    if goto_signup:
        ss["page"] = "signup"
        do_rerun()

    if submit_login:
        user = get_user_by_login(identifier.strip())
        if not user:
            st.error("User not found.")
            return
        if not verify_password(password, user["pwd_hash"], user["salt"]):
            st.error("Invalid credentials.")
            return

        ss["logged_in"] = True
        ss["user"] = {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
        }
        ss["page"] = "dashboard"
        if remember:
            save_persistent_session(user["id"], user["pwd_hash"])
        do_rerun()


def page_signup():
    ss = st.session_state
    scroll_top()

    st.markdown(
        """
<style>
div.block-container{
  min-height: 100vh !important;
  display: grid !important;
  place-items: center !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    with st.form("signup_form"):
        st.markdown(
            '<h3 class="gradient-heading" style="text-align:center;margin:0 0 10px 0;">Create your account</h3>',
            unsafe_allow_html=True,
        )
        username = st.text_input("Username", key="su_username", placeholder="Username")
        email = st.text_input("Email", key="su_email", placeholder="Email")
        pw1 = st.text_input(
            "Password", key="su_pw1", type="password", placeholder="Password"
        )
        pw2 = st.text_input(
            "Confirm Password",
            key="su_pw2",
            type="password",
            placeholder="Confirm Password",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            submit_signup = st.form_submit_button(
                "Create Account", width="stretch"
            )
        with c2:
            back_login = st.form_submit_button(
                "Back to Login", width="stretch"
            )

    if back_login:
        ss["page"] = "login"
        do_rerun()

    if submit_signup:
        if not username or not email or not pw1:
            st.error("Please fill all fields.")
            return
        if pw1 != pw2:
            st.error("Passwords do not match.")
            return
        ok, message = create_user(username.strip(), email.strip(), pw1)
        if ok:
            flash("success", "Account created. Please sign in.")
            ss["page"] = "login"
            do_rerun()
        else:
            st.error(str(message))


def reset_block_container_normal():
    st.markdown(
        """
<style>
div.block-container{
  min-height: unset !important;
  display: block !important;
  padding-top: 1rem !important;
  padding-bottom: 1.5rem !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def page_dashboard():
    ss = st.session_state
    reset_block_container_normal()
    needs_rerun = False  # For OAK auto-refresh

    # Navbar
    st.markdown('<div class="dashboard-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("Dashboard"):
        pass
    if c2.button("History"):
        ss["page"] = "history"
        do_rerun()
    if c3.button("Logout"):
        close_oak_device(ss)
        ss["logged_in"] = False
        ss["user"] = None
        ss["page"] = "login"
        clear_persistent_session()
        do_rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Hero
    st.markdown(
        '<div class="dashboard-wrap">'
        '<div class="card hero-card">'
        "<h1>Rebar Counting</h1>"
        "<p>Experience seamless live monitoring.</p>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Controls
    st.markdown(
        '<div class="dashboard-wrap"><span class="card-marker"></span></div>',
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown(
            "<h3 class='gradient-heading' style='margin:0 0 8px 0;'>Live Controls</h3>",
            unsafe_allow_html=True,
        )

        # Source selector
        current_idx = 0 if ss["source"] == "ip" else 1
        source_choice = st.radio(
            "Feed Source",
            options=["IP Webcam", "OAK-D Pro"],
            horizontal=True,
            index=current_idx,
            key="source_radio",
        )
        prev_source = ss["source"]
        ss["source"] = "ip" if source_choice == "IP Webcam" else "oak"
        if prev_source != ss["source"]:
            ss["streaming"] = False
            if prev_source == "oak":
                close_oak_device(ss)

        if ss["source"] == "ip":
            c1, c2, c3, c4, c5 = st.columns([1.9, 1.9, 1.2, 0.9, 0.9])
            with c1:
                ss["video_url"] = st.text_input(
                    "Video Stream URL (MJPEG)", ss.get("video_url", "")
                )
            with c2:
                ss["snapshot_url"] = st.text_input(
                    "Snapshot URL (JPEG)", ss.get("snapshot_url", "")
                )
            with c3:
                st.slider(
                    "Live Width (px)",
                    min_value=720,
                    max_value=1920,
                    value=int(ss.get("live_width", 1080)),
                    step=10,
                    key="live_width",
                )
            with c4:
                if st.button("Start Live"):
                    ss["streaming"] = True
            with c5:
                if st.button("Stop Live"):
                    ss["streaming"] = False
        else:
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.slider(
                    "Live Width (px)",
                    min_value=320,
                    max_value=1000,
                    value=int(ss.get("live_width_oak", 640)),
                    step=10,
                    key="live_width_oak",
                )
            with c2:
                if st.button("Start Live"):
                    ss["streaming"] = True
                    # Re-init device fresh on next frame
                    close_oak_device(ss)
            with c3:
                if st.button("Stop Live"):
                    ss["streaming"] = False
                    close_oak_device(ss)

        status_html = (
            '<span class="status-chip on">Live: ON</span>'
            if ss.get("streaming")
            else '<span class="status-chip off">Live: OFF</span>'
        )
        st.markdown(
            f'<div class="status-row">{status_html}</div>', unsafe_allow_html=True
        )

    # Live viewer
    live_width_key = "live_width" if ss["source"] == "ip" else "live_width_oak"
    live_width = int(ss.get(live_width_key, 640))
    if ss["streaming"]:
        if ss["source"] == "ip":
            mjpeg_url = ss.get("video_url", "")
            live_inner = f"""
              <img
                src="{mjpeg_url}"
                class="live-img"
                alt="Live stream IP"
              />
            """
        else:
            frame = None
            try:
                pipeline, queue = ensure_oak_device(ss)
                frame = get_oak_frame(pipeline, queue)
            except Exception as e:
                st.error(f"Failed to initialize OAK-D Pro: {e}")
                ss["streaming"] = False

            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                if w > live_width:
                    new_h = int(h * (live_width / w))
                    frame_rgb = cv2.resize(
                        frame_rgb, (live_width, new_h), interpolation=cv2.INTER_AREA
                    )
                ok, buf = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                )
                if ok:
                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    data_uri = "data:image/jpeg;base64," + b64
                    live_inner = (
                        f'<img src="{data_uri}" class="live-img" alt="Live OAK-D Pro" />'
                    )
                else:
                    live_inner = '<div style="padding:18px;color:#cbd5e1;text-align:center;">Failed to encode frame.</div>'
            else:
                live_inner = '<div style="padding:18px;color:#cbd5e1;text-align:center;">Waiting for OAK-D Pro frame...</div>'

        st.markdown(
            f"""
            <div class="dashboard-wrap">
              <div class="card live-card">
                <div class="live-wrap">
                  <div class="live-frame" style="width:{live_width}px; max-width:96vw;">
                    <div class="live-inner">{live_inner}</div>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if ss["source"] == "oak":
            needs_rerun = True
    else:
        if ss["source"] == "oak":
            close_oak_device(ss)
        st.markdown(
            """
            <div class="dashboard-wrap">
              <div class="card live-card">
                <div class="live-wrap">
                  <div style="padding:18px;color:#cbd5e1;text-align:center;">
                    Live stream is OFF. Click Start Live.
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Capture & Count + Upload
    st.markdown(
        '<div class="dashboard-wrap"><span class="card-marker"></span></div>',
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown(
            "<h3 class='gradient-heading' style='margin:0 0 8px 0;'>Capture & Count</h3>",
            unsafe_allow_html=True,
        )

        tab_source, tab_upload = st.tabs(["From Source", "Upload Image"])

        # From live source
        with tab_source:
            cap = st.button(
                "Capture & Count", width="stretch", key="cap_from_source"
            )
            if cap:
                if ss["source"] == "ip":
                    img_bgr, err = fetch_snapshot(ss.get("snapshot_url", ""))
                    stream_url = ss.get("video_url", "")
                    snapshot_url = ss.get("snapshot_url", "")
                    if err:
                        st.error(err)
                    else:
                        processed_rgb, count, derr = detect_rebars(
                            img_bgr, model, conf=0.6, iou=0.5, max_det=10000
                        )
                        if derr:
                            st.error(derr)
                        else:
                            det_id = record_detection(
                                ss["user"]["id"],
                                processed_rgb,
                                count,
                                stream_url,
                                snapshot_url,
                            )
                            ss["captured_images"].append(
                                {"id": det_id, "image": processed_rgb, "count": count}
                            )
                            MAX_CAP = 48
                            if len(ss["captured_images"]) > MAX_CAP:
                                ss["captured_images"] = ss["captured_images"][-MAX_CAP:]
                            st.success(f"Captured from IP. Detected rebars: {count}")
                            show_image_full_width(
                                processed_rgb,
                                caption=f"Detected rebars: {count}",
                            )
                else:
                    frame, oerr = grab_oak_frame(ss, wait_sec=2.0)
                    if oerr:
                        st.error(oerr)
                    else:
                        processed_rgb, count, derr = detect_rebars(
                            frame, model, conf=0.6, iou=0.5, max_det=10000
                        )
                        if derr:
                            st.error(derr)
                        else:
                            det_id = record_detection(
                                ss["user"]["id"],
                                processed_rgb,
                                count,
                                "OAK-D Pro",
                                "OAK-D Pro",
                            )
                            ss["captured_images"].append(
                                {"id": det_id, "image": processed_rgb, "count": count}
                            )
                            MAX_CAP = 48
                            if len(ss["captured_images"]) > MAX_CAP:
                                ss["captured_images"] = ss["captured_images"][-MAX_CAP:]
                            st.success(
                                f"Captured from OAK-D Pro. Detected rebars: {count}"
                            )
                            show_image_full_width(
                                processed_rgb,
                                caption=f"Detected rebars: {count}",
                            )

        # Upload Image tab
        with tab_upload:
            up_col1, up_col2 = st.columns([3, 1])
            with up_col1:
                uploaded_file = st.file_uploader(
                    "Upload image (JPG/PNG)",
                    type=["jpg", "jpeg", "png"],
                    key="uploader",
                )
            with up_col2:
                run_upload = st.button(
                    "Detect Uploaded Image",
                    width="stretch",
                    key="detect_upload",
                )

            if run_upload:
                if not uploaded_file:
                    st.warning("Please upload an image.")
                else:
                    file_bytes = np.frombuffer(
                        uploaded_file.getvalue(), np.uint8
                    )
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        st.error("Could not decode the uploaded image.")
                    else:
                        processed_rgb, count, derr = detect_rebars(
                            img_bgr, model, conf=0.6, iou=0.5, max_det=10000
                        )
                        if derr:
                            st.error(derr)
                        else:
                            det_id = record_detection(
                                ss["user"]["id"],
                                processed_rgb,
                                count,
                                "Upload",
                                uploaded_file.name or "Uploaded Image",
                            )
                            ss["captured_images"].append(
                                {"id": det_id, "image": processed_rgb, "count": count}
                            )
                            MAX_CAP = 48
                            if len(ss["captured_images"]) > MAX_CAP:
                                ss["captured_images"] = ss["captured_images"][-MAX_CAP:]
                            st.success(
                                f"Uploaded image processed. Detected rebars: {count}"
                            )
                            show_image_full_width(
                                processed_rgb,
                                caption=f"Detected rebars: {count}",
                            )

    # Gallery
    if ss["captured_images"]:
        cards = []
        for item in reversed(ss["captured_images"]):
            data_uri = img_to_data_uri(
                item["image"], quality=88, max_w=720
            )
            if not data_uri:
                continue
            cards.append(
                '<div class="shot-card">'
                f'<span class="shot-badge">Counts: {item["count"]}</span>'
                f'<img src="{data_uri}" alt="Detection"/>'
                f'<div class="shot-footer">Detected rebars: {item["count"]}</div>'
                "</div>"
            )
        gallery_html = '<div class="shot-grid">' + "".join(cards) + "</div>"
    else:
        gallery_html = (
            '<div style="color:#d7e2ff;">No captures yet. '
            "Click “Capture & Count” or upload an image.</div>"
        )

    st.markdown(
        '<div class="dashboard-wrap">'
        f'<div class="card gallery-card">{gallery_html}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Auto-refresh for OAK live view
    if needs_rerun:
        time.sleep(0.08)
        do_rerun()


def page_history():
    ss = st.session_state
    reset_block_container_normal()

    # Navbar
    st.markdown('<div class="dashboard-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("Dashboard"):
        ss["page"] = "dashboard"
        do_rerun()
    if c2.button("History"):
        pass
    if c3.button("Logout"):
        close_oak_device(ss)
        ss["logged_in"] = False
        ss["user"] = None
        ss["page"] = "login"
        clear_persistent_session()
        do_rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

    # History table
    st.markdown(
        '<div class="dashboard-wrap"><span class="card-marker"></span></div>',
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown(
            "<h2 class='gradient-heading' style='margin:0 0 6px 0;'>Detection History Records</h2>",
            unsafe_allow_html=True,
        )
        st.caption("Comprehensive log of all captured detections.")

        rows, total = list_detections(ss["user"]["id"], ss["hist_page"], PER_PAGE)
        total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)

        st.markdown('<div class="table-like">', unsafe_allow_html=True)
        col_widths = [0.8, 1.6, 2.6, 0.9, 0.6, 1.2]

        hcols = st.columns(col_widths, gap="small")
        hcols[0].markdown('<div class="th">ID</div>', unsafe_allow_html=True)
        hcols[1].markdown('<div class="th">Timestamp</div>', unsafe_allow_html=True)
        hcols[2].markdown('<div class="th">Stream URL</div>', unsafe_allow_html=True)
        hcols[3].markdown('<div class="th">Snapshot</div>', unsafe_allow_html=True)
        hcols[4].markdown('<div class="th">Count</div>', unsafe_allow_html=True)
        hcols[5].markdown('<div class="th">Actions</div>', unsafe_allow_html=True)

        base_index = (ss["hist_page"] - 1) * PER_PAGE

        for idx, r in enumerate(rows, start=1):
            serial_desc = total - (base_index + idx) + 1
            serial_label = f"ID {serial_desc:02d}"
            local_time = fmt_local_time(r["timestamp"])
            thumb_uri = file_to_data_uri(r["thumb_path"], max_w=90)

            rcols = st.columns(col_widths, gap="small")
            rcols[0].markdown(
                f'<div class="td">{serial_label}</div>', unsafe_allow_html=True
            )
            rcols[1].markdown(
                f'<div class="td">{local_time}</div>', unsafe_allow_html=True
            )
            rcols[2].markdown(
                f'<div class="td td-url">{r["stream_url"]}</div>',
                unsafe_allow_html=True,
            )
            if thumb_uri:
                rcols[3].markdown(
                    f'<div class="td"><img src="{thumb_uri}" '
                    'style="width:70px;height:auto;border-radius:8px;'
                    'border:1px solid rgba(255,255,255,0.15);" /></div>',
                    unsafe_allow_html=True,
                )
            else:
                rcols[3].markdown('<div class="td"></div>', unsafe_allow_html=True)
            rcols[4].markdown(
                f'<div class="td">{r["count"]}</div>', unsafe_allow_html=True
            )

            with rcols[5]:
                _wrap_top = st.markdown('<div class="td">', unsafe_allow_html=True)
                cc1, cc2 = st.columns([1, 1])
                with cc1:
                    if st.button("View", key=f"view_{r['id']}"):
                        if os.path.exists(r["image_path"]):
                            img = cv2.imread(r["image_path"])
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            show_image_full_width(
                                img_rgb,
                                caption=f"Detection {r['id']} — Count: {r['count']}",
                            )
                with cc2:
                    if st.button("Delete", key=f"del_{r['id']}"):
                         if delete_detection(r["id"], ss["user"]["id"]):
                            st.success("Deleted.")
                            do_rerun()
                _wrap_bottom = st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        prev_col, pages_col, next_col = st.columns([1, 3, 1])
        with prev_col:
            if st.button("Previous") and ss["hist_page"] > 1:
                ss["hist_page"] -= 1
                do_rerun()
        with pages_col:
            chips = []
            for p in range(1, total_pages + 1):
                cls = "page-chip active" if p == ss["hist_page"] else "page-chip"
                chips.append(f'<span class="{cls}">{p}</span>')
            st.markdown(
                '<div class="pagination">' + "".join(chips) + "</div>",
                unsafe_allow_html=True,
            )
        with next_col:
            if st.button("Next") and ss["hist_page"] < total_pages:
                ss["hist_page"] += 1
                do_rerun()
