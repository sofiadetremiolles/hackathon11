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
    Hsection_3 = st.container()

    # Prepare dataset for viewing
    dfRecos = st.session_state['stock_availability']
    dfProducts = st.session_state.uploaded_data['Products']
    dfRecos['ProductID'] = dfRecos['ProductID'].astype(str)
    dfProducts['ProductID'] = dfProducts['ProductID'].astype(str)
    dfMerged = dfRecos.merge(dfProducts[['ProductID', 'FamilyLevel2', 'Universe']], on='ProductID', how='left')
    dfMerged['Product_Label'] = dfMerged['FamilyLevel2'] + " (" + dfMerged['Universe'] + ")"
    dfRecos_Positive = dfMerged[dfMerged['recommendation_stock_diff'] > 0] # SKUs where potential recommendations outnumber stock

    # Prepare output CSV
    df_recos_exploded = st.session_state['recos'].explode('recommended_products')[['CustomerID','recommended_products']]
    df_recos_exploded['CustomerID'] = df_recos_exploded['CustomerID'].astype(str)
    df_recos_exploded['recommended_products'] = df_recos_exploded['recommended_products'].astype(str)
    df_recos_2 = df_recos_exploded.merge(dfProducts, left_on='recommended_products', right_on='ProductID', how='left')
    df_recos_2['Product_Name'] = df_recos_2['FamilyLevel2'] + " (" + df_recos_2['Universe'] + ")"
    df_recos_3 = df_recos_2.groupby('CustomerID').agg({
        'recommended_products': lambda x: ', '.join(map(str, x)),
        'Product_Name': lambda x: ', '.join(x)
    }).reset_index()
    df_recos_3.columns = ['CustomerID', 'Recommended Products', 'Product Names']

    
    with Hsection_1:
        st.markdown("<style>div.block-container {padding-top: 2rem;}</style>", unsafe_allow_html=True)
        st.subheader("Campaign Analysis")
   
        col1, col2 = st.columns(2)

        with col1:
            
            top_reco_products = df_recos_2.groupby(['ProductID', 'Product_Name']).size().reset_index(name='Count')
            top10_reco_products = top_reco_products.sort_values(by='Count', ascending=False).head(10)

            fig = px.bar(top10_reco_products, 
                        x='Count', 
                        y='Product_Name', 
                        orientation='h',
                        title='Current Generated Recommendations (Top 10 SKUs)',
                        color_discrete_sequence=['#ff4b4b'],
                        labels ={
                            "Count": "No. of Customers Recommended To",
                            "Product_Name": "SKU Name"},
                        text=top10_reco_products['Count'].apply(lambda x: f"{x}"),
                        )
            
            fig.update_traces(marker_line_width=1.5,
                  textfont=dict(color="white")  # Set data labels (text) color to white
                 )
            fig.update_layout(
                yaxis={
                    'categoryorder': 'total descending',  # Order categories by total value
                    'autorange': 'reversed'              # Reverse the order for descending display
                },
                height=310,
                margin=dict(l=20, r=20, t=40, b=20),
                bargap=0.15,
                legend=dict(
                    orientation="h",  # Horizontal orientation
                    yanchor="top",  
                    y=-0.25,  
                    xanchor="center",
                    x=0.5,
                    title_text = None
                )
            )

            for trace in fig.data:
                if trace.name == 'stocks_available':
                    trace.name = 'Current Reccomendations (based on Stock)'
                    trace.textfont = dict(color='grey')
                elif trace.name == 'Count':
                    trace.textfont = dict(color='white')
                
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:

            dfRecos_Positive['Sort_column'] = dfRecos_Positive['recommendation_stock_diff'] + dfMerged['stocks_available']
            dfRecos_Positive_Top10 = dfRecos_Positive.sort_values(by='Sort_column', ascending=False).head(10)
            dfPlot = pd.melt(dfRecos_Positive_Top10, id_vars=['Product_Label'], value_vars=['stocks_available', 'recommendation_stock_diff'],
                            var_name='Stock Type', value_name='Quantity')

            fig = px.bar(dfPlot, 
                        x='Quantity', 
                        y='Product_Label', 
                        color='Stock Type', 
                        color_discrete_map={'stocks_available': 'grey', 'recommendation_stock_diff': '#ff4b4b'},
                        orientation='h',
                        title='Additional Recommendations without Stock Limitations  (Top 10 SKUs)',
                        labels ={
                            "Quantity": "No. of Customers Reached",
                            "Product_Label": "SKU Name"},
                        text=dfPlot['Quantity'].apply(lambda x: f"+{x}" if x > 0 else str(x)),
                        )
            
            fig.update_traces(marker_line_width=1.5)
            fig.update_layout(
                yaxis={
                    'categoryorder': 'total descending',  # Order categories by total value
                    'autorange': 'reversed'              # Reverse the order for descending display
                },
                height=360,
                margin=dict(l=20, r=20, t=40, b=20),
                bargap=0.15,
                legend=dict(
                    orientation="h",  # Horizontal orientation
                    yanchor="top",  
                    y=-0.25,  
                    xanchor="center",
                    x=0.5,
                    title_text = None
                )
            )

            for trace in fig.data:
                if trace.name == 'stocks_available':
                    trace.name = 'Current Reccomendations (based on Stock)'
                    trace.textfont = dict(color='grey')
                elif trace.name == 'recommendation_stock_diff':
                    trace.name = 'Potential Reccomendations'
                    trace.textfont = dict(color='white')
                
            
            st.plotly_chart(fig, use_container_width=True)
            
            
    with Hsection_3:
        st.subheader("Export Generated Recommendations")

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(df_recos_3)

        # Download button
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name='data.csv',
            mime='text/csv',
        )

        st.write(df_recos_3)


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

