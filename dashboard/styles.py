import streamlit as st

def apply_theme():
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
            padding: 20px;
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
        },

        [data-testid="stSidebar"] button {
            color: #259c36 !important;  /* Color for links and buttons */
        },
        
        .stMultiSelect div[data-baseweb="select"] div[data-selected="true"] {
        background-color: #259C36 !important;  /* Set selected item background */
        color: white !important;  /* Set text color for selected items */
    }

    </style>
    """,
    unsafe_allow_html=True
    )