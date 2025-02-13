import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pipeline import whole_pipeline
import plotly.express as px
import ast
import matplotlib.pyplot as plt

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




def show():

    Hsection_0 = st.container()
    Hsection_1 = st.container()
    Hsection_2 = st.container()
   

    with Hsection_0:
        # Upload model outputs
        output_files = st.file_uploader("Test Files (Model Outputs)", type=["csv"], key='uploadoutputfiles', 
                                        accept_multiple_files=True, label_visibility= 'collapsed')
        
        if 'output_files' not in st.session_state:
            st.session_state.output_files = {}
        
        if output_files:
            for uploaded_file in output_files:
                file_name = uploaded_file.name
                if "recos" in file_name.lower():
                    file_label = "recos_test"
                elif "stock_availability" in file_name.lower():
                    file_label = "stock_availability_test"
                elif "transactions" in file_name.lower():
                    file_label = "transactions"
                elif "products" in file_name.lower():
                    file_label = "products"
                else:
                    file_label = "Unknown"
        
                st.session_state.output_files[file_label] = pd.read_csv(uploaded_file)
        
        st.write(st.session_state.output_files['recos_test'])
        st.write(st.session_state.output_files['stock_availability_test'])
    
    with Hsection_1:


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
        
        results_file = map_recommendations(st.session_state.output_files['products'], st.session_state.output_files['recos_test'])
        
        st.write(results_file)

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(results_file)

        # Download button
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='data.csv',
            mime='text/csv',
        )
    



        
 



st.markdown('</div>', unsafe_allow_html=True)

