import json
import os
import sys

import duckdb
import pandas as pd

# Add workspace root to sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

PREFIX = 'gpt_columns' #options: 'our', 'claude', 'gpt', 'gpt_columns'

TRANSACTION_DATA_FILE_PATH = 'data/Store_Sales_Price_Elasticity_Promotions_Data.parquet'
DATASET_FILE_PATH = f"evaluation/{PREFIX}_dataset.json"
table_name = "sales"

queries = {
    'our': [
        "What was the most popular product SKU?",
        "What was the total revenue across all stores?",
        "Which store had the highest sales volume?",
        "Create a bar chart showing total sales by store",
        "What was the average transaction value?",
        "Retrieve the sales made in November 2021 and December 2021 and tell me when more money were earned?",
        "Show me the sales in Nov 2021",
        "Collect sales data for December 2021 and tell which day had the highest sales?",
        "How many products were sold for a promo in May 2023?",
        "Weekly sales in 2021",
        "Extract the first and last sale in time order"
    ],
    'claude': [
        "What are the total sales and quantity sold for each store in 2023?",
        "Which 20 SKUs had the highest total sales value between January 2022 and December 2022?",
        "How many sales transactions occurred in each month of 2023?",
        "What is the average sale value for each product class code that has more than 100 transactions?",
        "Show all sales from store number 1980 where the quantity sold was greater than 5",
        "List all sales that were on promotion in 2024",
        "Which stores made sales with a total sale value above $100 in Q1 2022?",
        "What are the top 15 stores by total quantity sold in 2022?",
        "Show the daily total sales value for store 2970 in February 2024",
        "Which 25 product class codes had the most promotional sales in 2023?"
    ],
    'gpt_columns' : [
        "Return the 12 months of 2023 with total revenue and total units sold; columns: month_start, total_revenue, total_units; order by month_start ASC.",
        "Return the top 20 SKUs by total units sold across the full dataset; tie-break by higher total revenue, then SKU_Coded ASC; columns: SKU_Coded, total_units, total_revenue.",
        "For 2022, return the top 15 Product_Class_Code by total revenue; tie-break by higher total units, then Product_Class_Code ASC; columns: Product_Class_Code, total_revenue, total_units.",
        "For 2023, return the top 25 stores by promotional unit share (promo_units/total_units) among stores with at least 100 total units; tie-break by Store_Number ASC; columns: Store_Number, promo_unit_share, promo_units, total_units.",
        "For Q1 2023 (2023-01-01 to 2023-03-31), return the top 30 stores by total revenue; tie-break by Store_Number ASC; columns: Store_Number, total_revenue, total_units.",
        "For 2023, return the top 10 SKUs by total revenue with avg unit price = sum(Total_Sale_Value)/sum(Qty_Sold); tie-break by SKU_Coded ASC; columns: SKU_Coded, total_revenue, total_units, avg_unit_price.",
        "For 2023, return month-by-promo aggregates with columns: month_start, On_Promo, total_revenue, total_units; order by month_start ASC, On_Promo ASC; expect 24 rows.",
        "For 2023, return the top 50 stores by distinct SKUs sold; tie-break by Store_Number ASC; columns: Store_Number, distinct_skus.",
        "For 2023, return the top 30 Product_Class_Code by promotional units sold (On_Promo=1); tie-break by Product_Class_Code ASC; columns: Product_Class_Code, promo_units.",
        "Across the full dataset, return the 40 SKUs with the highest average unit price among SKUs with at least 50 units sold; avg_unit_price = sum(Total_Sale_Value)/sum(Qty_Sold); tie-break by SKU_Coded ASC; columns: SKU_Coded, avg_unit_price, total_units, total_revenue.",
        "Return the 30 most recent Sold_Date values aggregated by date with columns: Sold_Date, total_revenue, total_units; order by Sold_Date DESC.",
        "For 2022, return the top 25 stores by total units sold for Product_Class_Code = 22975; tie-break by Store_Number ASC; columns: Store_Number, total_units, total_revenue.",
    ],
    'gpt' : [
        "Return the 12 months of 2023 with total revenue and total units sold; order by month_start ASC.",
        "Return the top 20 SKUs by total units sold across the full dataset; tie-break by higher total revenue, then SKU_Coded ASC.",
        "For 2022, return the top 15 Product_Class_Code by total revenue; tie-break by higher total units, then Product_Class_Code ASC.",
        "For 2023, return the top 25 stores by promotional unit share (promo_units/total_units) among stores with at least 100 total units; tie-break by Store_Number ASC.",
        "For Q1 2023 (2023-01-01 to 2023-03-31), return the top 30 stores by total revenue; tie-break by Store_Number ASC.",
        "For 2023, return the top 10 SKUs by total revenue with avg unit price = sum(Total_Sale_Value)/sum(Qty_Sold); tie-break by SKU_Coded ASC.",
        "For 2023, return month-by-promo aggregates; order by month_start ASC, On_Promo ASC; expect 24 rows.",
        "For 2023, return the top 50 stores by distinct SKUs sold; tie-break by Store_Number ASC.",
        "For 2023, return the top 30 Product_Class_Code by promotional units sold (On_Promo=1); tie-break by Product_Class_Code ASC.",
        "Across the full dataset, return the 40 SKUs with the highest average unit price among SKUs with at least 50 units sold; avg_unit_price = sum(Total_Sale_Value)/sum(Qty_Sold); tie-break by SKU_Coded ASC.",
        "Return the 30 most recent Sold_Date values aggregated by date; order by Sold_Date DESC.",
        "For 2022, return the top 25 stores by total units sold for Product_Class_Code = 22975; tie-break by Store_Number ASC."
    ]
}


df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

SQL_GENERATION_PROMPT = """Generate an SQL query based on the prompt.
Please just reply with the SQL query and NO MORE, just the query.
The prompt is : {prompt}. The available columns are: {columns}. The table name is: {table_name}.
If you need to use a DATE column with LIKE or pattern matching, first CAST it to VARCHAR like this: CAST(date_column AS VARCHAR) LIKE '%2021-11%'.
Return only the SQL query, with no explanations or markdown formatting.
Format your response as a single continuous line with no line breaks.
"""

DATA_ANALYSIS_PROMPT = """Your goal is to give a clear answer to this question: {prompt}.
Use the information available and return only a direct answer to the question.
The only data available is the data extracted from another agent using this SQL code: {sql_query}
The output of the SQL you can use to answer is this data: {data}
Format your response as a single continuous line with no line breaks, using semicolons or commas to separate items.
"""

dataset = []

# Loop over all questions and ask user to input SQL and analysis for each
for i, prompt in enumerate(queries[PREFIX]):
    if any(entry['prompt'] == prompt for entry in dataset):
        print(f'\nSkipping question {i+1}/{len(queries[PREFIX])} as it already exists in the dataset.')
        continue
    formatted_prompt = SQL_GENERATION_PROMPT.format(prompt=prompt, columns=df.columns.to_list(), table_name=table_name)
    
    print(f'\n{"="*80}')
    print(f'Question {i+1}/{len(queries[PREFIX])}: {prompt}')
    print('\nPrompt for LLM:')
    print(f'{"="*80}')
    print(formatted_prompt)
    print('-'*80)
    
    # Ask user to input the SQL query
    user_sql = input('\nPlease enter the SQL gt query for this question: ')

    # Data extraction part
    sql_query = user_sql.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "")
    
    try:
        result = duckdb.sql(sql_query).df()
        data = result.to_string()
    except Exception as e:
        print(f"\nError executing query: {e}")
    
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(
        data=data, prompt=prompt, sql_query=sql_query
    )

    print('\nRequest for LLM:')
    print(f'{"="*80}')
    print(formatted_prompt)
    print('-'*80)

    # Ask user to input the text analysis
    text_analysis = input('\nPlease enter the gt text for this question: ')

    # Save to dataset
    dataset.append({
        'prompt': prompt,
        'gt_sql': sql_query,
        'gt_data': data,
        'gt_analysis': text_analysis
    })

# Save dataset as json file
with open(f"evaluation/{PREFIX}_dataset.json", 'w') as f:
    json.dump(dataset, f, indent=2)

print(f'\n{"="*80}')
print('Dataset completed and saved!')
print(f'Total entries: {len(dataset)}')
print(f'{"="*80}')
