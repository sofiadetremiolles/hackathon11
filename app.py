import streamlit as st
import pandas as pd
import os
#from model_pipeline import run_model
import utils

import streamlit as st
import pandas as pd
import plotly.express as px

# Set up Streamlit page
st.set_page_config(page_title="Transaction Data Analysis", layout="wide")

# Initialize session state for data storage
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}

st.title("Transaction Data Explorer")

# **Step 1: Upload Files and Store in Session State**
st.header("Upload Transaction Data")
uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])

if uploaded_file:
    st.session_state.uploaded_data["Transactions"] = pd.read_csv(uploaded_file)
    st.success("File uploaded and stored in session!")

# **Step 2: Create a Visualization**
if "Transactions" in st.session_state.uploaded_data:
    transactions = st.session_state.uploaded_data["Transactions"]
    
    st.subheader("Transactions Overview")
    
    # Select a numeric column to visualize
    numeric_columns = transactions.select_dtypes(include=["number"]).columns
    if len(numeric_columns) > 0:
        selected_column = st.selectbox("Select a column to visualize", numeric_columns)
        
        # Create a histogram
        fig = px.histogram(transactions, x=selected_column, title=f"Distribution of {selected_column}")
        st.plotly_chart(fig)
    else:
        st.warning("No numeric columns found in the uploaded file.")

# **Step 3: Aggregate Transactions Data**
def aggregate_transactions(df):
    """Aggregate transaction data by Customer ID, summing total purchases."""
    if "ClientID" in df.columns and "SalesNetAmountEuro" in df.columns:
        agg_df = df.groupby("ClientID", as_index=False)["SalesNetAmountEuro"].sum()
        agg_df.rename(columns={"SalesNetAmountEuro": "TotalSpent"}, inplace=True)
        return agg_df
    else:
        st.error("Required columns 'CustomerID' and 'TransactionAmount' not found.")
        return None

if "Transactions" in st.session_state.uploaded_data:
    st.subheader("Aggregated Transactions")
    aggregated_df = aggregate_transactions(transactions)
    
    if aggregated_df is not None:
        st.dataframe(aggregated_df)
        
        # **Step 4: Download Aggregated Data**
        csv = aggregated_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Aggregated Transactions CSV",
            data=csv,
            file_name="aggregated_transactions.csv",
            mime="text/csv",
        )