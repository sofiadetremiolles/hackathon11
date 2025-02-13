import pandas as pd
import numpy as np
from collections import Counter, defaultdict

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

# Inputs

#transactions = pd.read_csv("data/transactions.csv")
#products = pd.read_csv("data/products.csv")
#clients = pd.read_csv("data/clients.csv")
#stores = pd.read_csv("data/stores.csv")
#stocks = pd.read_csv("data/stocks.csv")
conversion_rate = 0.01
number_of_recommandations = 3
end_date = '2024-11-01'

# Pipeline Master Function
def whole_pipeline(transactions, clients, products, stores, stocks, end_date, number_of_recommandations, conversion_rate):
    
    transactions_final = transform_to_transactions_final(transactions, clients, products, stores, stocks)
    clusters = clustering(transactions, clients, products)
    training_data, test_data, test_data_customers_all_purchased = preprocessing(transactions, clients, products, stores, stocks, clusters, end_date)
    matrix_proba = pairing_and_training(training_data, test_data)
    recos = optimize_recommendations_improved(transactions_final, matrix_proba, number_of_recommandations,conversion_rate )
    grouped_transactions = filter_and_group_transactions(transactions_final, recos)
    recall_per_customer, overall_recall = compute_recall(recos, grouped_transactions)
    top_products = get_top_k_products(matrix_proba, number_of_recommandations)
    stock_availability = count_product_recommendations(top_products, stocks)
    
    return clusters, matrix_proba, recos, overall_recall, stock_availability

# Clustering
"""
Input:
products.csv, transactions.csv, clients.csv
Output:
A partition of users into 19 clusters
"""

def get_client_category(filtered: pd.DataFrame, number_mode: int) -> pd.DataFrame:
    filtered = filtered.sort_values(by=["ClientID", "SaleTransactionDate"], ascending=[True, True])

    result = []

    for client_id, group in filtered.groupby("ClientID"):
        last_n_transactions = group.tail(number_mode)
        categories = last_n_transactions["Category"].tolist()

        if len(categories) == 1:
            chosen_category = categories[0]
        else:

            counter = Counter(categories)
            most_common = counter.most_common()

            max_count = most_common[0][1]
            tied_categories = [cat for cat, count in most_common if count == max_count]

            if len(tied_categories) == 1:
                chosen_category = tied_categories[0]
            else:

                for cat in reversed(categories):
                    if cat in tied_categories:
                        chosen_category = cat
                        break

        result.append((client_id, chosen_category))

    return pd.DataFrame(result, columns=["ClientID", "Category"])


def clustering(transactions, clients, products):

    data = pd.merge(transactions, clients, on='ClientID', how='left')
    data = pd.merge(data, products, on='ProductID', how='left')

    # Trier par ClientID et SaleTransactionDate
    data = data.sort_values(by=['ClientID', 'SaleTransactionDate'])
    data = data[data["ClientCountry"] == "FRA"]

    columns_to_keep = ["ClientID","SaleTransactionDate","Category"]
    filtered = data.loc[:, data.columns.isin(columns_to_keep)]
    clusters = get_client_category(filtered, 3)
    return clusters


# Preprocessing

def transform_to_transactions_final(transactions, clients, products, stores, stocks):
    df_stores = stores
    df_stocks = stocks
    df_clients = clients
    df_products = products
    df_transactions = transactions

    df_transactions=df_transactions.drop_duplicates()
    df_stocks = df_stocks.rename(columns={'Quantity': 'Stock_Quantity'})
    df_transactions = df_transactions.merge(df_stores[['StoreID', 'StoreCountry']],
                                      on='StoreID',
                                      how='left')
    df_transactions = df_transactions.merge(df_stocks[['StoreCountry', 'ProductID', 'Stock_Quantity']],
                                        on=['StoreCountry', 'ProductID'],
                                        how='left')
    df_transactions = df_transactions.merge(df_clients,
                                        on='ClientID',
                                        how='left')
    df_products['Brand'] = df_products['FamilyLevel2'].str.extract(r'^(\S+)')
    df_transactions = df_transactions.merge(df_products,
                                        on='ProductID',
                                        how='left')
    return df_transactions


def create_train_test(df, end_date='2024-11-01', n_train_week=13, n_predict_week=2):
    # Convert end_date to datetime if it's not already
    end_date = pd.to_datetime(end_date).tz_localize(None)  # Make end_date timezone-naive
    df['SaleTransactionDate'] = pd.to_datetime(df['SaleTransactionDate']).dt.tz_localize(None) # Make SaleTransactionDate timezone-naive

    # Calculate the start and end dates for the desired range
    start_date_test = end_date - pd.DateOffset(weeks=n_predict_week)
    start_date_train = start_date_test - pd.DateOffset(weeks=n_train_week)

    # Filter the dataframe
    train_df = df[(df['SaleTransactionDate'] >= start_date_train) & (df['SaleTransactionDate'] < start_date_test)]
    test_df = df[(df['SaleTransactionDate'] >= start_date_test) & (df['SaleTransactionDate'] < end_date)]

    return train_df, test_df

def process_data(dfTxn):
    # Date and Quarter Info
    dfTxn['SaleTransactionDate'] = pd.to_datetime(dfTxn['SaleTransactionDate'])
    dfTxn['SaleTransactionQtr'] = dfTxn['SaleTransactionDate'].dt.to_period('Q')
    ''' NEED TO CHECK WHY REMOVED A BIT'''
    dfTxn['SaleTransactionWeek'] = dfTxn['SaleTransactionDate'].dt.strftime('%Y-%U')
    dfTxn['SaleTransactionQtr'].unique()

    dfTxn_Merged = dfTxn[(dfTxn['ClientCountry']=='FRA')]
    football_clients = clusters.loc[clusters["Category"] == "Football", "ClientID"]
    # Filtrage de dfTxn_Merged pour ne garder que les transactions des clients identifiés
    dfTxn_Merged = dfTxn_Merged[dfTxn_Merged["ClientID"].isin(football_clients)]
    # Assume the segment is Category = "Football"
    Keep_for_test = dfTxn_Merged
    dfTxn_Merged = dfTxn_Merged[dfTxn_Merged['Category']=="Football"]

    # Unit Price
    dfAgg_UnitPrice = dfTxn_Merged[['ProductID', 'SalesNetAmountEuro', 'Quantity']]\
        .groupby('ProductID', as_index= False).agg( {'SalesNetAmountEuro' : 'sum', 'Quantity' : 'sum'})

    dfAgg_UnitPrice['UnitPrice'] = dfAgg_UnitPrice['SalesNetAmountEuro']/dfAgg_UnitPrice['Quantity']


    # Product Purchase Frequency

    dfAgg_ProductFreq = dfTxn_Merged.groupby('ProductID', as_index=False)\
        .agg(OrderCount=('ProductID', 'count'))


    # No. of Store Visits
    dfTxn_Merged['SaleTransactionDate'] = pd.to_datetime(dfTxn_Merged['SaleTransactionDate'])
    dfAgg_StoreVisits = dfTxn_Merged.groupby('ClientID', as_index = False).agg(VisitCount = ('SaleTransactionDate', lambda x : x.dt.date.nunique()))


    # No. of Past Transactions, Median order value, Unique SKUs purchased
    dfAgg_Customer = dfTxn_Merged.groupby('ClientID', as_index= False).agg(
        TxnCount = ('ClientID', 'count'),
        MedianOrderValue = ('SalesNetAmountEuro', 'median'),
        SKUCount = ('ProductID', 'nunique')
    )

    # Customer x Product features

    dfCustomer_x_Brand = dfTxn_Merged.groupby(['ClientID', 'Brand'], as_index= False).agg({'SalesNetAmountEuro' : 'sum'})
    dfCustomer_x_ProdCatL1 = dfTxn_Merged.groupby(['ClientID', 'FamilyLevel1'], as_index= False).agg({'SalesNetAmountEuro' : 'sum'})
    dfCustomerTotalSpend  = dfTxn_Merged.groupby('ClientID').agg(Customer_TotalSales = ('SalesNetAmountEuro' , 'sum'))

    dfCustomer_x_Brand = dfCustomer_x_Brand.merge(dfCustomerTotalSpend, on = 'ClientID')
    dfCustomer_x_ProdCatL1 = dfCustomer_x_ProdCatL1.merge(dfCustomerTotalSpend, on = 'ClientID')

    dfCustomer_x_Brand['%_byBrand'] = dfCustomer_x_Brand['SalesNetAmountEuro']/dfCustomer_x_Brand['Customer_TotalSales']
    dfCustomer_x_ProdCatL1['%_byProdCatL1'] = dfCustomer_x_ProdCatL1['SalesNetAmountEuro']/dfCustomer_x_ProdCatL1['Customer_TotalSales']

    dfPivot_Customer_x_Brand = dfCustomer_x_Brand.pivot(index = 'ClientID', columns = 'Brand', values = '%_byBrand').fillna(0)
    dfPivot_Customer_x_ProdCatL1 = dfCustomer_x_ProdCatL1.pivot(index = 'ClientID', columns = 'FamilyLevel1', values = '%_byProdCatL1').fillna(0)

    # Encoded columns - Product
    dfProduct_info = dfTxn_Merged[['ProductID', 'FamilyLevel1', 'Brand']].drop_duplicates()
    dfProduct_info = pd.get_dummies(dfProduct_info, columns=['FamilyLevel1', 'Brand'], drop_first= True)
    dfProduct_info = dfProduct_info.astype('int')

    # Encoded columns - Customer
    dfCustomer_info = dfTxn_Merged[['ClientID', 'ClientCountry', 'ClientSegment']].drop_duplicates()
    dfCustomer_info = pd.get_dummies(dfCustomer_info, columns=['ClientCountry', 'ClientSegment'], drop_first= True)
    dfCustomer_info = dfCustomer_info.astype('int')

    # Final Dataset for XGBoost

    unique_customers = dfTxn_Merged['ClientID'].unique()
    unique_products = dfTxn_Merged['ProductID'].unique()
    CustomerSKU_Cartesian = pd.MultiIndex.from_product([unique_customers, unique_products], names=['ClientID', 'ProductID'])

    dfCustomerSKU = pd.DataFrame(index=CustomerSKU_Cartesian).reset_index()

    # Flag column if Customer purchased SKU
    dfCustomerSKU_Flag = dfTxn_Merged[['ClientID', 'ProductID']].groupby(['ClientID','ProductID'], as_index= False).apply(lambda x: x.assign(PurchasedFlag=1))

    dfCustomerSKU = pd.merge(dfCustomerSKU, dfCustomerSKU_Flag, on = ['ClientID', 'ProductID'], how = 'left')

    # Join other aggregated tables
    dfCustomerSKU = pd.merge(dfCustomerSKU, dfAgg_UnitPrice, on='ProductID', how='left')
    dfCustomerSKU = pd.merge(dfCustomerSKU, dfAgg_ProductFreq, on='ProductID', how='left')
    dfCustomerSKU = pd.merge(dfCustomerSKU, dfAgg_StoreVisits, on='ClientID', how='left')
    dfCustomerSKU = pd.merge(dfCustomerSKU, dfAgg_Customer, on='ClientID', how='left')
    dfCustomerSKU = pd.merge(dfCustomerSKU, dfPivot_Customer_x_Brand, on='ClientID', how='left')
    dfCustomerSKU = pd.merge(dfCustomerSKU, dfPivot_Customer_x_ProdCatL1, on='ClientID', how='left')
    dfCustomerSKU = pd.merge(dfCustomerSKU, dfCustomer_info, on='ClientID', how='left')

    dfCustomerSKU.fillna(0, inplace=True)

    # Export
    dfCustomerSKU.drop_duplicates(inplace= True)
    return dfCustomerSKU, Keep_for_test



def preprocessing(transactions, clients, products, stores, stocks, clusters, end_date):
    #Returns the feature engineering of the training_data, along with the test_data
    #Should also resturn the test data for the football cluster but containing all purchases (also outside of the category) => a list of product ID bought by each.
    transactions_final = transform_to_transactions_final(transactions, clients, products, stores, stocks)

    dfTxn = transactions_final
    dfTxn = dfTxn.sort_values(by=['SaleTransactionDate'])
    df_train, df_test = create_train_test(dfTxn, end_date)

    training_data, _ = process_data(df_train)
    test_data, test_data_customers_all_purchased = process_data(df_test)

    return training_data, test_data, test_data_customers_all_purchased


# Note: the test_data_customers_all_purchased is the list of transactions made by the people clustered inside football regardless of the category

# Model Training
"""
input: training_data, test_data \
output: probabilities matrices for both test (for metrics computation) and global train+test (for the overall recommandations)
"""

def twist_recommend(df_pred_info_merged, recommend_recent=False):
    if not recommend_recent:
        df_pred_info_merged.loc[df_pred_info_merged['PurchasedFlag_prev'] == 1, 'PurchasedFlag_pred'] = 0
    return df_pred_info_merged

def pairing_and_training(df_train, df_test):
    #pairs both training_data and test_data clients/products separately.
    #Trains the model only on the training_data
    #makes predictions for training data and the test data => the prediction on the training set will allow to have the full matrix of recommandations, while the prediction on the test set gives us the recall score
    ID_cols = ['ClientID','ProductID']
    features = ['UnitPrice','OrderCount','VisitCount','TxnCount','MedianOrderValue','SKUCount','Ball','Jersey', 'Shoes', 'Shorts', 'ClientSegment_LOYAL',
        'ClientSegment_TOP']
    X_train = df_train[features + ID_cols]
    y_train = df_train[['PurchasedFlag'] + ID_cols]
    X_test = df_test[features + ID_cols]
    y_test = df_test[['PurchasedFlag']+ID_cols]
    # Initialize and train the XGBoost Regressor
    model = xgb.XGBClassifier(objective='binary:logistic')

    model.fit(X_train.drop(ID_cols, axis=1), y_train.drop(ID_cols, axis=1))
    y_proba = model.predict_proba(X_test.drop(ID_cols, axis=1))[:, 1]


    df_pred_info = pd.DataFrame({'ClientID':X_test['ClientID'],'ProductID':X_test['ProductID'],'PurchasedFlag_truth':y_test['PurchasedFlag'],'PurchasedFlag_pred':y_proba})
    y_prev = y_train.copy()
    y_prev.rename(columns={'PurchasedFlag':'PurchasedFlag_prev'}, inplace=True)

    df_pred_info_merged=df_pred_info.merge(y_prev,on=['ClientID','ProductID'],how='left')

    df_pred_info_merged = twist_recommend(df_pred_info_merged,recommend_recent = False)
    matrix_proba_test = df_pred_info_merged.pivot(index='ClientID', columns='ProductID', values='PurchasedFlag_pred')
    matrix_proba_test = matrix_proba_test.reset_index()

    return matrix_proba_test



# Optimization and Recommandations Generation
"""
input: test_probabilities, stocks, stores, conversion_rate

output: matrix of recommandations
"""

def optimize_recommendations_improved(df, probability_matrix, k, lambda_param):

    # Convert ProductIDs to strings in stock levels
    df = df.dropna(how='all')
    df=df[(df['ClientCountry'] == 'FRA') & (df['StoreCountry'] == 'FRA') & (df['Category'] == 'Football')]
    stock_levels = df.groupby('ProductID')['Stock_Quantity'].first()
    stock_levels.index = stock_levels.index.astype(str)
    stock_levels = stock_levels.to_dict()

    customers = probability_matrix['ClientID'].unique()
    recommendations = []

    # Configure solver with better parameters
    solver = PULP_CBC_CMD(msg=False, timeLimit=60, gapRel=0.1)

    for customer in customers:
        customer_probs = probability_matrix[probability_matrix['ClientID'] == customer].iloc[0]
        customer_probs = customer_probs.drop('ClientID')

        # Track available products before stock check
        products_before_stock = len(customer_probs.index)

        available_products = [
            product for product in customer_probs.index
            if stock_levels.get(str(product), 0) > 0
        ]


        if len(available_products) == 0:
            continue

        # Create optimization problem
        prob = LpProblem(f"Customer_{customer}_Recommendation", LpMaximize)

        # Create variables only for available products
        X = {
            product: LpVariable(f'X_{customer}_{product}', cat='Binary')
            for product in available_products
        }

        if len(X) > 0:
            # Objective: Maximize probability-weighted recommendations
            prob += lpSum(X[i] * customer_probs[i] for i in X)

            # Relaxed constraints for number of recommendations
            prob += lpSum(X[i] for i in X) <= min(k, len(X))
            #prob += lpSum(X[i] for i in X) >= min(k-1, len(X))  # Allow k-1 recommendations

            # Stock constraints with adjusted lambda
            for product in available_products:
                prob += lambda_param * lpSum(X[product]) <= stock_levels[str(product)]

            # Solve with improved parameters
            status = prob.solve(solver)

            if status == 1:  # Optimal solution found
                selected_products = [p for p in X if X[p].varValue == 1]

                # Update stock levels using original lambda_param
                for product in selected_products:
                    stock_levels[str(product)] = max(0, stock_levels[str(product)] - lambda_param)

                # Store recommendations
                recommendations.append({
                    'CustomerID': customer,
                    'recommended_products': selected_products,
                    'recommendation_scores': [customer_probs[p] for p in selected_products]
                })


    result_df = pd.DataFrame(recommendations)

    return result_df


# Metrics
"""
input: test_recommandations, test_data \
output: score
"""

def filter_and_group_transactions(transactions_final, recos, end_date='2024-11-01', n_predict_week=2):
    # Convert end_date to datetime and remove timezone
    end_date = pd.to_datetime(end_date).tz_localize(None)
    transactions_final['SaleTransactionDate'] = pd.to_datetime(transactions_final['SaleTransactionDate']).dt.tz_localize(None)

    # Calculate start date for the last two weeks
    start_date_test = end_date - pd.DateOffset(weeks=n_predict_week)

    # Filter transactions based on the last two weeks
    filtered_transactions = transactions_final[
        (transactions_final['SaleTransactionDate'] >= start_date_test) &
        (transactions_final['SaleTransactionDate'] < end_date)
    ]

    # Filter transactions to keep only ClientIDs that appear in recos
    filtered_transactions = filtered_transactions[
        filtered_transactions['ClientID'].isin(recos['CustomerID'])
    ]

    # Group by ClientID and collect unique ProductID as lists
    grouped_transactions = filtered_transactions.groupby('ClientID')['ProductID'].unique().apply(list).reset_index()

    return grouped_transactions


def compute_recall(recos, grouped_transactions):
    """
    Computes the recall of recommendations.

    Parameters:
    - recos: DataFrame with 'CustomerID' and 'recommended_products' (list of recommended products per customer).
    - grouped_transactions: DataFrame with 'ClientID' and 'ProductID' (list of purchased products per customer).

    Returns:
    - recall_per_customer: Dictionary with recall per customer.
    - overall_recall: Overall recall score.
    """

    # Convert to dictionaries for quick lookup
    recos_dict = recos.set_index('CustomerID')['recommended_products'].to_dict()
    transactions_dict = grouped_transactions.set_index('ClientID')['ProductID'].to_dict()

    recall_per_customer = {}
    total_recommended = 0
    total_bought = 0

    for client in transactions_dict.keys():
        bought_products = set(transactions_dict.get(client, []))
        recommended_products = set(recos_dict.get(client, []))

        if len(bought_products) > 0:
            correctly_recommended = len(bought_products & recommended_products)
            recall = correctly_recommended / len(bought_products)
            recall_per_customer[client] = recall

            total_recommended += correctly_recommended
            total_bought += len(bought_products)

    # Compute overall recall
    overall_recall = total_recommended / total_bought if total_bought > 0 else 0

    return recall_per_customer, overall_recall


# Compute ratios out of stock
"""
On output:\
Un tableau with ProductID, WantedToRecommend, Stocks, Difference, Ratio
"""


def get_top_k_products(probability_matrix, k):
    # Ensure ClientID is set as index if not already
    if 'ClientID' in probability_matrix.columns:
        probability_matrix = probability_matrix.set_index('ClientID')

    # Get the top-k ProductIDs for each ClientID
    top_k_products = probability_matrix.apply(lambda row: row.nlargest(k).index.tolist(), axis=1)

    # Reset index to keep ClientID as a column and return
    return top_k_products.reset_index(name='top_k_products')



def count_product_recommendations(top_k_recommendations):
    """
    Counts the number of times each ProductID has been recommended.

    Parameters:
    - top_k_recommendations: DataFrame with 'ClientID' and 'top_k_products' (list of recommended ProductIDs per customer).

    Returns:
    - A DataFrame with 'ProductID' and 'times_recommended'.
    """

    # Flatten the list of lists into a single list of all recommended ProductIDs
    all_recommended_products = [product for sublist in top_k_recommendations['top_k_products'] for product in sublist]

    # Count occurrences of each ProductID
    product_counts = pd.Series(all_recommended_products).value_counts().reset_index()
    product_counts.columns = ['ProductID', 'times_recommended']

    return product_counts



def count_product_recommendations(top_k_recommendations, stocks):

    # Flatten the list of lists into a single list of all recommended ProductIDs
    all_recommended_products = [product for sublist in top_k_recommendations['top_k_products'] for product in sublist]

    # Count occurrences of each ProductID
    product_counts = pd.Series(all_recommended_products).value_counts().reset_index()
    product_counts.columns = ['ProductID', 'times_recommended']

    # Get stocks available in France (FRA)
    stocks_fra = stocks[stocks['StoreCountry'] == 'FRA'][['ProductID', 'Quantity']]
    stocks_fra = stocks_fra.rename(columns={'Quantity': 'stocks_available'})

    # Merge stock availability with product recommendation counts
    merged_df = product_counts.merge(stocks_fra, on='ProductID', how='left')

    # Fill missing stock values with 0 (assuming no stock in France if missing)
    merged_df['stocks_available'] = merged_df['stocks_available'].fillna(0)

    # Compute difference and ratio
    merged_df['recommendation_stock_diff'] = merged_df['times_recommended'] - merged_df['stocks_available']
    merged_df['recommendation_stock_ratio'] = merged_df['times_recommended'] / merged_df['stocks_available'].replace(0, 1)  # Avoid division by zero
    merged_df = merged_df.sort_values(by='recommendation_stock_ratio', ascending=False)


    return merged_df






