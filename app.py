import streamlit as st

# Layout
st.set_page_config(
    page_title="Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded")

from dashboard.styles import apply_theme
from streamlit_option_menu import option_menu
from dashboard import dash_page1, dash_page2

# Theme
apply_theme()

#Options Menu
with st.sidebar:
    selected = option_menu('Your Next Purchase',
                           ["Campaign Settings", 'Campaign Overview'], 
        icons=['gear','eye'],menu_icon='cart', default_index=0
        
    )

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}

if selected == "Campaign Settings":
    st.markdown('<div class="config-box">', unsafe_allow_html=True)
    dash_page1.show()

if selected == "Campaign Overview":
    st.markdown('<div class="config-box">', unsafe_allow_html=True)
    
    if 'clusters' in st.session_state:
        dash_page2.show()
    else:
        st.markdown("<span style='color: grey'>Generate a campaign to view outputs.</span>", unsafe_allow_html=True)



    


