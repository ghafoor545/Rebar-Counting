import streamlit as st

from db import init_db
from style import inject_global_css
from auth import load_persistent_session, get_user_by_id
from pages import page_login, page_signup, page_dashboard, page_history


def init_session():
    ss = st.session_state
    ss.setdefault("logged_in", False)
    ss.setdefault("user", None)
    ss.setdefault("page", "login")
    ss.setdefault("streaming", False)
    ss.setdefault("captured_images", [])
    ss.setdefault("hist_page", 1)
    ss.setdefault("source", "ip")  # "ip" or "oak"
    ss.setdefault("live_width", 1080)
    ss.setdefault("live_width_oak", 640)
    ss.setdefault("video_url", "http://10.4.32.132:8080/video")
    ss.setdefault("snapshot_url", "http://10.4.32.132:8080/shot.jpg")
    ss.setdefault("oak_device", None)
    ss.setdefault("oak_queue", None)


def auto_login_if_token():
    ss = st.session_state
    if not ss["logged_in"]:
        uid = load_persistent_session()
        if uid is not None:
            u = get_user_by_id(uid)
            if u:
                ss["logged_in"] = True
                ss["user"] = {
                    "id": u["id"],
                    "username": u["username"],
                    "email": u["email"],
                }
                ss["page"] = "dashboard"


def main():
    st.set_page_config(page_title="Rebar Counting", page_icon="Rebar", layout="wide")

    inject_global_css()
    init_db()
    init_session()
    auto_login_if_token()

    ss = st.session_state

    if not ss["logged_in"]:
        if ss["page"] == "signup":
            page_signup()
        else:
            page_login()
        return

    if ss.get("page") == "history":
        page_history()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()