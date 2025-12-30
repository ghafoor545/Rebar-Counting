import streamlit as st
from streamlit.components.v1 import html as st_html


def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        try:
            from streamlit.runtime.scriptrunner import RerunException, RerunData
            raise RerunException(RerunData(None))
        except Exception:
            pass


def scroll_top():
    st_html("<script>window.scrollTo(0,0);</script>", height=0)


def show_image_full_width(img, caption=None):
    st.image(img, caption=caption, width="stretch")


def flash(kind: str, text: str):
    st.session_state["_flash"] = (kind, str(text))


def show_flash():
    f = st.session_state.pop("_flash", None)
    if not f:
        return
    kind, text = f
    if kind == "success":
        st.success(str(text))
    elif kind == "error":
        st.error(str(text))
    elif kind == "warning":
        st.warning(str(text))
    else:
        st.info(str(text))
