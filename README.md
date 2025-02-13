# hackathon11
Code for the DSBA Eleven Strategy Hackathon 

# Marketing Campaign Planner (Proof of Concept)

## Overview
This project provides a **Streamlit-based platform** for planning marketing campaigns with **personalized recommendations** for customers. Users can set campaign parameters, generate recommendations, and access insights on campaign performance. 

**Note:** This is a **Proof of Concept (PoC)** with limited functionality.

## Installation
### Prerequisites
- Python (>=3.8)
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```

### Running the App
Run the following command in your terminal:
```sh
streamlit run app.py
```
**Important:** The app may initially show a `streamlit.errors.StreamlitDuplicateElementId` error. Simply refresh the page to proceed.

## Features
Once the app is running, you will have access to three tabs:
1. **Campaign Settings** – Upload data and configure campaign parameters.
2. **Campaign Overview** – View campaign details.
3. **Campaign Performance** – Analyze campaign results.

## How to Run a Campaign
### Step 1: Upload Data
Click **“Browse Files”** and upload the following CSV files:
- `transactions.csv`
- `stocks.csv`
- `stores.csv`
- `clients.csv`
- `products.csv`

### Step 2: Configure Parameters
Adjust the following settings:
- **Number of Recommendations per Customer**
- **Recommendation Conversion Rate (%)**
  - This defines the expected **conversion rate** (probability of purchase per recommendation).
  - Example:
    - If set to **100% (1.0)** → Each recommended product is expected to be sold. Thus, recommendations cannot exceed stock availability.
    - If set to **10% (0.1)** → Each product is expected to be sold **once per 10 recommendations**, meaning we can recommend it up to **10 times the available stock**.

⚠️ **Country and Customer Type settings are hardcoded and do not affect recommendations.**

### Step 3: Generate Campaign
Click **“Generate Campaign”**. Once you see **“Campaign Generated Successfully!”**, navigate to other tabs for insights.

### Step 4: Restarting a Campaign
To launch a new campaign:
- **Refresh the page** and **re-upload data** before generating a new campaign.
- Skipping this step may cause issues.

## License
This project is a **personal proof of concept** intended for evaluation purposes only. It is not licensed for public distribution, modification, or commercial use.

## Contact
For questions or contributions, open an issue in this repository.

