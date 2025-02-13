import streamlit as st

#Layout
st.set_page_config(
    page_title="Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded")

from streamlit_option_menu import option_menu
from dashboard import dash_page1 

#from urllib.request import urlopen

#import plotly.express as px
#import plotly.graph_objects as go

#import json
#import requests
#import pydeck as pdk




st.markdown(
    """
    <style>
        /* Custom large font */
        .big-font {
            font-size: 80px !important;
        },

        /* Custom config box styling */
        .config-box {
            background-color: #f0f2f6;  /* Light gray background */
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin-top: -20px;
            margin-bottom: 20px;
        },

        /* Override option menu font to match Streamlit default */
        .css-1v0mbdj, .css-16huue1, .css-1aehpvj {  
            font-family: "Source Sans Pro", sans-serif !important;
        }

        /* Padding between colums */
        [data-testid="stHorizontalBlock"] > div {
        padding-right: 15px;
        }

    </style>
    """,
    unsafe_allow_html=True
)


#Options Menu
with st.sidebar:
    selected = option_menu('Your Next Purchase',
                           ["Campaign Settings", 'Campaign Overview','Campaign Performance'], 
        icons=['gear','eye','bar-chart'],menu_icon='cart', default_index=0)

if selected == "Campaign Settings":
    st.markdown('<div class="config-box">', unsafe_allow_html=True)
    dash_page1.show()

if selected == "Campaign Overview":
    st.markdown('<div class="config-box">', unsafe_allow_html=True)
    st.title("Recommender System Outputs")
    
    if 'clusters' in st.session_state:
        st.subheader("Clusters")
        st.write(st.session_state['clusters'])

        st.subheader("Probability Matrix")
        st.write(st.session_state['matrix_proba'])

        st.subheader("Recommendations")
        st.write(st.session_state['recos'])

        st.subheader("Overall Recall")
        st.write(f"Overall Recall: {st.session_state['overall_recall']}")

        st.subheader("Stock Availability")
        st.write(st.session_state['stock_availability'])

    


