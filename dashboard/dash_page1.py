import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pipeline import whole_pipeline
from dashboard.styles import apply_theme

apply_theme()

#Options Menu
with st.sidebar:
    selected = option_menu('Your Next Purchase',
                           ["Campaign Settings", 'Campaign Overview','Campaign Performance'], 
        icons=['gear','eye','bar-chart'],menu_icon='cart', default_index=0)


def show():

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
            if 'required_files' not in st.session_state:
                st.session_state.required_files = file_labels
            
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
            k = st.slider("Number of Recommendations per Customer", 1, 15, 3, key = 'kSelector')
            st.session_state.k = int(k)
        
        with col2_4:
            lambda_value = st.number_input("Recommendation Conversion Rate (%)", min_value=0.01, max_value=1.0, value=0.5, step=0.1, key='lambdaSelector')
            st.session_state.lambda_value = float(lambda_value)

    with Hsection_4:
        st.subheader("Generate Campaign")

        if st.button('Generate'):
            
            # Store input CSVs as dataframes
            if all(st.session_state.upload_status.get(label, False) for label in st.session_state.required_files):
                transactions = st.session_state.uploaded_data["Transactions"]
                products = st.session_state.uploaded_data["Products"]
                clients = st.session_state.uploaded_data["Clients"]
                stocks = st.session_state.uploaded_data["Stocks"]
                stores = st.session_state.uploaded_data["Stores"]
            
            # Run the Pipeline
            # note: def whole_pipeline(transactions, clients, products, stores, stocks, end_date, number_of_recommandations, conversion_rate):
            
            clusters, matrix_proba, recos, overall_recall, stock_availability = whole_pipeline(
                transactions, clients, products, stores, stocks, 
                end_date= '2024-11-01',
                number_of_recommandations = st.session_state.k,
                conversion_rate = st.session_state.lambda_value
            )

            # Store the outputs in the session state
            if 'clusters' not in st.session_state:
                st.session_state['clusters'] = clusters
                st.session_state['matrix_proba'] = matrix_proba
                st.session_state['recos'] = recos
                st.session_state['overall_recall'] = overall_recall
                st.session_state['stock_availability'] = stock_availability
            
            st.write("Campaign Generated Successfully!")

st.markdown('</div>', unsafe_allow_html=True)

