import streamlit as st
from oled_display import oled_show_ready
from db import init_db
from style import inject_global_css
from auth import load_persistent_session, get_user_by_id
from pages import page_login, page_signup, page_dashboard, page_history


def init_session():
    """Initialize session state variables"""
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
    ss.setdefault("oled_ready_shown", False)
    ss.setdefault("oled_available", False)


def auto_login_if_token():
    """Automatically log in user if valid session token exists"""
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


def is_raspberry_pi():
    """Check if running on Raspberry Pi"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            if 'Raspberry Pi' in f.read():
                return True
    except:
        pass
    return False


def main():
    """Main application entry point"""
    # Configure page
    st.set_page_config(
        page_title="Rebar Counting",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize application components
    inject_global_css()
    init_db()
    init_session()
    auto_login_if_token()
    
    ss = st.session_state
    
    # Show "Rebar App / Ready" once on OLED display
    if not ss.get("oled_ready_shown", False):
        # Only attempt OLED display on Raspberry Pi
        if is_raspberry_pi():
            try:
                oled_show_ready()
                ss["oled_available"] = True
                ss["oled_ready_shown"] = True
            except ImportError:
                # OLED libraries not installed
                ss["oled_available"] = False
                ss["oled_ready_shown"] = True
            except Exception as e:
                # OLED hardware not connected or other error
                import traceback
                print(f"OLED display error: {e}")
                traceback.print_exc()
                ss["oled_available"] = False
                ss["oled_ready_shown"] = True
        else:
            # Not running on Raspberry Pi
            ss["oled_available"] = False
            ss["oled_ready_shown"] = True
    
    # Check authentication status and route to appropriate page
    if not ss["logged_in"]:
        if ss["page"] == "signup":
            page_signup()
        else:
            page_login()
        return
    
    # User is logged in, show appropriate page
    if ss.get("page") == "history":
        page_history()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
