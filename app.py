import streamlit as st
import pandas as pd
from urllib.request import urlopen

import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json
import requests
import pydeck as pdk

from dashboard import dash_page1 

#Layout
st.set_page_config(
    page_title="Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded")

#Data Pull and Functions

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

def my_function():
    a = 2 + 3

#Options Menu
with st.sidebar:
    selected = option_menu('Your Next Purchase',
                           ["Campaign Settings", 'Campaign Overview','Campaign Performance'], 
        icons=['gear','eye','bar-chart'],menu_icon='cart', default_index=0)

if selected == "Campaign Settings":
    dash_page1.show()

if selected == "Campaign Overview":
    st.title("Overview here")








