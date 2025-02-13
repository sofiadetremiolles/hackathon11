import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pipeline import whole_pipeline
import plotly.express as px
import ast
import matplotlib.pyplot as plt
from dashboard.styles import apply_theme

apply_theme()


def show():

    Hsection_1 = st.container()
    Hsection_2 = st.container()

    
    with Hsection_1:

        st.subheader("Top Product Recommendations")
   
        if "uploaded_data" in st.session_state and "recos" in st.session_state:
            st.session_state.results_file = map_recommendations(st.session_state.uploaded_data['Products'], st.session_state['recos'])
        
    with Hsection_2:

        st.subheader("Export Recommendations")

        if "results_file" in st.session_state:
            st.write(st.session_state.results_file)

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            
            toCSV = st.session_state.results_file
            csv = convert_df_to_csv(toCSV)

            # Download button
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='data.csv',
                mime='text/csv',
            )


# Function to display the bar chart
def map_recommendations(products_df, recos_df):
    # Explode the recommended products so each product is in a separate row
    recos_df = recos_df.copy()
    recos_df['recommended_products'] = recos_df['recommended_products'].apply(ast.literal_eval)  # Convert string to list
    recos_exploded = recos_df.explode('recommended_products')
    
    # Convert to numeric for merging
    recos_exploded['recommended_products'] = pd.to_numeric(recos_exploded['recommended_products'])
    products_df['ProductID'] = pd.to_numeric(products_df['ProductID'])
    
    # Merge to bring in FamilyLevel1 and FamilyLevel2
    merged_df = recos_exploded.merge(products_df, left_on='recommended_products', right_on='ProductID', how='left')
    
    # Create Product column using FamilyLevel1 and the first word of FamilyLevel2
    merged_df['Product'] = merged_df['FamilyLevel2'].str.split().str[0] + ' ' + merged_df['FamilyLevel1'] + ' ' + merged_df['Universe']
    
    # Select relevant columns
    result_df = merged_df[['CustomerID', 'recommended_products', 'Product']]
    
    # Create a horizontal bar chart showing the number of customers per product
    product_counts = result_df['Product'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    product_counts.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel("Number of Customers")
    ax.set_ylabel("Product")
    ax.set_title("Number of Customers Recommended Each Product")
    ax.invert_yaxis()

    # Display chart in Streamlit
    st.pyplot(fig)

    return result_df
 



st.markdown('</div>', unsafe_allow_html=True)

