import streamlit as st

from st_login_form import login_form
def set_background_image():
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("https://img.upanh.tv/2024/05/24/4-wxLLDdDYg-transformed.png");
        background-size: cover
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
set_background_image()
client = login_form()

if st.session_state["authenticated"]:
    if st.session_state["username"]:
        st.success(f"Welcome {st.session_state['username']}")
    else:
        st.success("Welcome guest")
else:
    st.error("Not authenticated")