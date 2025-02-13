import streamlit as st
import pandas as pd
from urllib.request import urlopen

import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json
import requests
import pydeck as pdk

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

Hsection_1 = st.container()
Hsection_2 = st.container()
Hsection_3 = st.container()
Hsection_4 = st.container()

# 1. Data Uploader
with Hsection_1:
    
    st.markdown('<div class="config-box">', unsafe_allow_html=True) 
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Input Data")
        if "uploaded_data" not in st.session_state:
            st.session_state.uploaded_data = {}
    
        file_labels = ["Transactions", "Products", "Clients", "Stocks", "Stores"]
        upload_status = {label: False for label in file_labels}

        uploaded_files = st.file_uploader("Upload Data Files", type=["csv"], key='uploadfiles', accept_multiple_files=True, label_visibility= 'collapsed')


        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                if "transactions" in file_name.lower():
                    file_label = "Transactions"
                elif "products" in file_name.lower():
                    file_label = "Products"
                elif "clients" in file_name.lower():
                    file_label = "Clients"
                elif "stocks" in file_name.lower():
                    file_label = "Stocks"
                elif "stores" in file_name.lower():
                    file_label = "Stores"
                else:
                    file_label = "Unknown"
        
                st.session_state.uploaded_data[file_label] = pd.read_csv(uploaded_file)
                upload_status[file_label] = True
        
        if upload_status is not None:
            st.session_state.upload_status = upload_status
               
# 2b. File upload status
with Hsection_2:
    cols = st.columns(len(file_labels)) 
    for idx, file_label in enumerate(file_labels):
        with cols[idx]:
            is_uploaded = upload_status.get(file_label, False)
            st.checkbox(f"{file_label}.csv", value=is_uploaded, disabled=True, key=f"{file_label}_status")

# 3. Campaign Settings     
with Hsection_3:
    st.subheader("Adjust Campaign Parameters")

    col2_1, col2_2, col2_3, col2_4 = st.columns(4)

    with col2_1:
        campaign_countries = st.multiselect(
            "Create campaign for countries", 
            ["FRA","USA","DEU","GBR","BRA","ARE","AUS"],
            ["FRA"],
            help= "<help text>",
            key = 'countryselector')

    with col2_2:
        campaign_custLabels = st.multiselect(
            "Create campaign for customer types", 
            ["LOYAL","TOP","PROSPECT","INACTIVE_1Y"],
            ["LOYAL","TOP","PROSPECT","INACTIVE_1Y"],
            help= "<help text>",
            key = 'custLabelselector')

    with col2_3:
        st.slider("Number of Recommendations per Customer", 1, 15, 3, key = 'kSelector')
    
    with col2_4:
        st.number_input("Recommendation Strength (%)", min_value=0.01, max_value=1.0, value=0.5, step=0.1, key='alphaselector')

with Hsection_4:
    st.subheader("Generate Campaign")
    if st.button('Generate'):
        result = my_function()
        st.write("Campaign Generated Successfully!")

st.markdown('</div>', unsafe_allow_html=True)







