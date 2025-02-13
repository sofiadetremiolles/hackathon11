import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pipeline import whole_pipeline
import plotly as px

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


def show():

    Hsection_1 = st.container()
   
   

    with Hsection_1:

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
                else:
                    file_label = "Unknown"
        
                st.session_state.output_files[file_label] = pd.read_csv(uploaded_file)
        
        st.write(st.session_state.output_files['recos_test'])
        
        def analyze_pre_post_opti(pre_df, post_df):
            """
            Streamlit version:
            - Extracts product recommendation counts from post-optimization data.
            - Merges it with pre-optimization data.
            - Compares pre vs. post-optimization results.
            - Visualizes the changes with an interactive horizontal bar chart using Plotly.
            """

            # Step 1: Extract Post-Optimization Product Counts
            post_df_exploded = post_df.explode('recommended_products')
            post_product_counts = post_df_exploded['recommended_products'].value_counts().reset_index()
            post_product_counts.columns = ['ProductID', 'Post_Count']

            # Step 2: Process Pre-Optimization Data
            pre_df = pre_df.iloc[:, :2]  # Keep only first two columns
            pre_df = pre_df.rename(columns={'ProductID': 'ProductID', 'times_recommended': 'Pre_Count'})

            # Ensure 'ProductID' columns in both DataFrames are of the same type (e.g., str)
            pre_df['ProductID'] = pre_df['ProductID'].astype(str)
            post_product_counts['ProductID'] = post_product_counts['ProductID'].astype(str)

            st.write(pre_df)
            st.write(post_product_counts)

            # Step 3: Merge Pre and Post Data
            merged_df = pd.merge(pre_df, post_product_counts, on="ProductID", how="left")  # 'inner' keeps common products

            st.write(merged_df)

            # Step 4: Calculate Customer Loss
            merged_df["Difference"] = merged_df["Pre_Count"] - merged_df["Post_Count"]

            # Step 5: Compute KPI (Average Customer Loss)
            avg_loss_per_product = merged_df["Difference"].mean()

            # Step 6: Sort by Pre_Count in Descending Order
            merged_df = merged_df.sort_values(by="Pre_Count", ascending=False)

            # Streamlit UI
            st.title("📊 Pre vs Post Optimization Analysis")

            # Show KPI
            st.metric("📉 Average Customer Loss per Product", f"{avg_loss_per_product:.2f}")

            # Show Dataframe
            st.subheader("📋 Merged Data Preview")
            st.dataframe(merged_df)

            # Step 7: Visualization - Horizontal Bar Chart using Plotly
            st.subheader("📊 Pre vs Post Optimization: Customer Recommendations per Product")
            
            fig = px.bar(
                merged_df,
                y="ProductID",
                x=["Pre_Count", "Post_Count"],
                orientation="h",
                barmode="group",
                title="Pre vs Post Optimization: Customer Recommendations per Product",
                labels={"value": "Number of Customers Recommended", "ProductID": "Product ID"},
                color_discrete_map={"Pre_Count": "#0D47A1", "Post_Count": "#64B5F6"},  # Dark Blue & Light Blue
            )
            
            fig.update_yaxes(categoryorder="total ascending")  # Ensures highest values on top
            st.plotly_chart(fig, use_container_width=True)

            return merged_df, avg_loss_per_product
        
        if "recos_test" in st.session_state.output_files:
            analyze_pre_post_opti(st.session_state.output_files['stock_availability_test'], st.session_state.output_files['recos_test'])
 


        

st.markdown('</div>', unsafe_allow_html=True)

