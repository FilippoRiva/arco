"""
Generate synthetic products.parquet and stores.parquet tables
that join with Store_Sales_Price_Elasticity_Promotions_Data.parquet.

Run once: py data/generate_synthetic_tables.py
"""
import numpy as np
import pandas as pd
import duckdb
import os

RNG = np.random.default_rng(seed=42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SALES_PATH = os.path.join(DATA_DIR, "../Store_Sales_Price_Elasticity_Promotions_Data.parquet")

# ── Load keys from existing sales data ──────────────────────────────────────
sales = duckdb.query(f"SELECT * FROM '{SALES_PATH}'").df()

store_numbers = sorted(sales["Store_Number"].unique())   # 35 stores
sku_class_df  = (
    sales[["SKU_Coded", "Product_Class_Code"]]
    .drop_duplicates()
    .sort_values("SKU_Coded")
    .reset_index(drop=True)
)  # 659 SKUs

# ── STORES TABLE ────────────────────────────────────────────────────────────
CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "Indianapolis", "San Francisco", "Seattle", "Denver", "Nashville",
    "Oklahoma City", "El Paso", "Washington", "Las Vegas", "Louisville",
    "Memphis", "Portland", "Baltimore", "Milwaukee", "Albuquerque",
    "Tucson", "Fresno", "Sacramento", "Kansas City", "Mesa",
]
REGIONS = {
    "New York": "Northeast",       "Los Angeles": "West",         "Chicago": "Midwest",
    "Houston": "South",            "Phoenix": "West",             "Philadelphia": "Northeast",
    "San Antonio": "South",        "San Diego": "West",           "Dallas": "South",
    "San Jose": "West",            "Austin": "South",             "Jacksonville": "South",
    "Fort Worth": "South",         "Columbus": "Midwest",         "Charlotte": "South",
    "Indianapolis": "Midwest",     "San Francisco": "West",       "Seattle": "West",
    "Denver": "West",              "Nashville": "South",          "Oklahoma City": "South",
    "El Paso": "South",            "Washington": "Northeast",     "Las Vegas": "West",
    "Louisville": "Midwest",       "Memphis": "South",            "Portland": "West",
    "Baltimore": "Northeast",      "Milwaukee": "Midwest",        "Albuquerque": "West",
    "Tucson": "West",              "Fresno": "West",              "Sacramento": "West",
    "Kansas City": "Midwest",      "Mesa": "West",
}
STORE_TYPES = ["Supermarket", "Hypermarket", "Discount", "Convenience"]
STORE_TYPE_WEIGHTS = [0.45, 0.25, 0.20, 0.10]

stores = pd.DataFrame({
    "Store_Number": store_numbers,
    "store_name": [f"FreshMart #{n}" for n in store_numbers],
    "city": CITIES[:len(store_numbers)],
    "region": [REGIONS[c] for c in CITIES[:len(store_numbers)]],
    "store_type": RNG.choice(STORE_TYPES, size=len(store_numbers), p=STORE_TYPE_WEIGHTS),
    "opening_year": RNG.integers(2000, 2020, size=len(store_numbers)),
    "area_sqm": RNG.integers(500, 5000, size=len(store_numbers)),
    "num_employees": RNG.integers(20, 300, size=len(store_numbers)),
    "has_parking": RNG.choice([True, False], size=len(store_numbers), p=[0.7, 0.3]),
})
stores["Store_Number"] = stores["Store_Number"].astype(int)

# ── PRODUCTS TABLE ───────────────────────────────────────────────────────────
# Product_Class_Code → category/subcategory mapping (based on code ranges)
CLASS_CATEGORY = {
    22800: ("Beverages",       "Soft Drinks"),
    22825: ("Beverages",       "Juices"),
    22850: ("Dairy",           "Milk & Cream"),
    22875: ("Dairy",           "Cheese"),
    22900: ("Snacks",          "Chips & Crisps"),
    22925: ("Snacks",          "Cookies & Biscuits"),
    22950: ("Bakery",          "Bread & Rolls"),
    22975: ("Bakery",          "Pastries"),
    24375: ("Household",       "Cleaning Supplies"),
    24400: ("Household",       "Paper Products"),
    24425: ("Personal Care",   "Hygiene & Beauty"),
}

BRANDS_BY_CATEGORY = {
    "Beverages":     ["CoolSip", "FizzUp", "AquaVita", "SunDrop", "PureFlow"],
    "Dairy":         ["CreamyFarm", "NaturalBest", "WhiteGold", "FreshValley", "AlpineSelect"],
    "Snacks":        ["CrunchTime", "SnackZone", "TastyBite", "MunchPack", "CrispLeaf"],
    "Bakery":        ["GoldenCrust", "BakeFresh", "HarvestBake", "DoughCo", "SoftBite"],
    "Household":     ["CleanHome", "SparkleUp", "FreshGuard", "EcoClean", "BrightCare"],
    "Personal Care": ["GlowUp", "SoftSkin", "PurePlus", "CareEssentials", "NaturalTouch"],
}

PRODUCT_NAMES = {
    ("Beverages",   "Soft Drinks"):        ["Cola Classic", "Lemon Fizz", "Orange Burst", "Grape Soda", "Ginger Ale"],
    ("Beverages",   "Juices"):             ["Apple Juice 1L", "Orange Juice 1L", "Mango Nectar", "Berry Mix", "Tropical Blend"],
    ("Dairy",       "Milk & Cream"):       ["Whole Milk 1L", "Skim Milk 1L", "Heavy Cream", "Half & Half", "Oat Drink"],
    ("Dairy",       "Cheese"):             ["Cheddar Slices", "Mozzarella Block", "Gouda Wheel", "Parmesan Grated", "Swiss Slices"],
    ("Snacks",      "Chips & Crisps"):     ["Sea Salt Chips", "BBQ Crisps", "Sour Cream Chips", "Tortilla Rounds", "Veggie Straws"],
    ("Snacks",      "Cookies & Biscuits"): ["Chocolate Chip", "Oat & Raisin", "Butter Biscuits", "Digestives", "Wafer Rolls"],
    ("Bakery",      "Bread & Rolls"):      ["White Sandwich Loaf", "Whole Grain Loaf", "Sourdough Round", "Brioche Bun", "Multigrain Roll"],
    ("Bakery",      "Pastries"):           ["Croissant", "Cinnamon Roll", "Blueberry Muffin", "Danish Pastry", "Pain au Chocolat"],
    ("Household",   "Cleaning Supplies"):  ["All-Purpose Spray", "Floor Cleaner", "Bathroom Gel", "Dish Soap", "Laundry Liquid"],
    ("Household",   "Paper Products"):     ["Kitchen Roll 3pk", "Toilet Paper 6pk", "Facial Tissues", "Paper Napkins", "Baking Paper"],
    ("Personal Care","Hygiene & Beauty"):  ["Shampoo 400ml", "Body Wash 500ml", "Deodorant Stick", "Hand Cream", "Face Wash"],
}

ORIGINS_BY_CATEGORY = {
    "Beverages":     ["USA", "Mexico", "Germany", "France", "Brazil"],
    "Dairy":         ["USA", "Netherlands", "France", "Ireland", "New Zealand"],
    "Snacks":        ["USA", "UK", "Mexico", "Netherlands", "Canada"],
    "Bakery":        ["USA", "France", "Italy", "Germany", "Canada"],
    "Household":     ["USA", "Germany", "UK", "Netherlands", "China"],
    "Personal Care": ["USA", "France", "South Korea", "Germany", "Japan"],
}

product_rows = []
for _, row in sku_class_df.iterrows():
    sku = int(row["SKU_Coded"])
    pcc = int(row["Product_Class_Code"])
    cat, subcat = CLASS_CATEGORY[pcc]
    brand_list = BRANDS_BY_CATEGORY[cat]
    name_list  = PRODUCT_NAMES[(cat, subcat)]
    origin_list = ORIGINS_BY_CATEGORY[cat]

    brand   = brand_list[sku % len(brand_list)]
    name    = name_list[sku % len(name_list)]
    origin  = origin_list[sku % len(origin_list)]
    # unit price: seeded by SKU for reproducibility
    rng_sku = np.random.default_rng(seed=sku)
    unit_price  = round(float(rng_sku.uniform(0.5, 25.0)), 2)
    weight_kg   = round(float(rng_sku.uniform(0.1, 3.0)), 3)
    is_organic  = bool(rng_sku.choice([True, False], p=[0.2, 0.8]))

    product_rows.append({
        "SKU_Coded":          sku,
        "Product_Class_Code": pcc,
        "category":           cat,
        "subcategory":        subcat,
        "product_name":       name,
        "brand":              brand,
        "unit_price_usd":     unit_price,
        "weight_kg":          weight_kg,
        "is_organic":         is_organic,
        "country_of_origin":  origin,
    })

products = pd.DataFrame(product_rows)

# ── Save to parquet ──────────────────────────────────────────────────────────
stores_path   = os.path.join(DATA_DIR, "../stores.parquet")
products_path = os.path.join(DATA_DIR, "../products.parquet")

stores.to_parquet(stores_path,   index=False)
products.to_parquet(products_path, index=False)

print(f"stores.parquet   → {len(stores)} rows, columns: {list(stores.columns)}")
print(f"products.parquet → {len(products)} rows, columns: {list(products.columns)}")

# ── Quick join sanity check ──────────────────────────────────────────────────
joined = duckdb.query(f"""
    SELECT s.Store_Number, st.city, st.region,
           p.brand, p.category,
           SUM(s.Total_Sale_Value) AS total_revenue
    FROM '{SALES_PATH}' s
    JOIN '{stores_path}' st ON s.Store_Number = st.Store_Number
    JOIN '{products_path}' p ON s.SKU_Coded = p.SKU_Coded
    WHERE YEAR(CAST(s.Sold_Date AS DATE)) = 2023
    GROUP BY 1, 2, 3, 4, 5
    ORDER BY total_revenue DESC
    LIMIT 5
""").df()
print("\nJoin sanity check (top 5 brand×region revenue 2023):")
print(joined.to_string(index=False))
