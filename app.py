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
                           ["Campaign Settings", 'Campaign Overview','##TEST PAGE'], 
        icons=['gear','eye','bar-chart'],menu_icon='cart', default_index=0
        
    )

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}

if selected == "Campaign Settings":
    st.markdown('<div class="config-box">', unsafe_allow_html=True)
    dash_page1.show()

if selected == "Campaign Overview":
    st.markdown('<div class="config-box">', unsafe_allow_html=True)
    
    #Temporary output
    if 'clusters' in st.session_state:
        st.subheader("Customer Segment Assignment")
        st.write(st.session_state['clusters'])

        st.subheader("Product Recommendations")
        st.write(st.session_state['recos'])

        st.subheader("Overall Recall")
        st.write(f"Overall Recall: {st.session_state['overall_recall']}")

if selected == "##TEST PAGE":
    st.markdown('<div class="config-box">', unsafe_allow_html=True)
    dash_page_test.show()


    


