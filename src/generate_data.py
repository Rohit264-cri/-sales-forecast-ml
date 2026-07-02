"""
generate_data.py  —  Generates 3 years of realistic retail sales data
Run: python src/generate_data.py
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_sales_data(output_path='data/sales_data.csv'):
    dates      = pd.date_range('2022-01-01', '2024-12-31', freq='W-MON')
    n          = len(dates)
    trend      = np.linspace(50000, 90000, n)
    seasonal   = 15000 * np.sin(2 * np.pi * np.arange(n) / 52 - np.pi/2)
    festival   = np.zeros(n)
    for i,d in enumerate(dates):
        if d.month == 10: festival[i] = 20000  # Diwali
        if d.month == 12: festival[i] = 18000  # Christmas
        if d.month ==  1: festival[i] =  8000  # New Year
    noise      = np.random.normal(0, 4000, n)
    sales      = trend + seasonal + festival + noise
    sales      = np.clip(sales, 10000, None)

    df = pd.DataFrame({
        'date':             dates,
        'sales':            np.round(sales, 2),
        'units_sold':       np.round(sales / np.random.uniform(45, 75, n)).astype(int),
        'marketing_spend':  np.round(sales * np.random.uniform(0.08, 0.14, n), 2),
        'discount_pct':     np.round(np.random.uniform(0, 25, n), 1),
        'temperature':      np.round(np.random.uniform(15, 42, n), 1),
        'is_holiday':       [1 if d.month in [10,12,1] else 0 for d in dates],
        'week_of_year':     [d.isocalendar()[1] for d in dates],
        'month':            [d.month for d in dates],
        'quarter':          [d.quarter for d in dates],
        'year':             [d.year for d in dates],
    })

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved    -> {output_path}")
    print(f"Rows             : {len(df)}")
    print(f"Date range       : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Avg Weekly Sales : Rs {df['sales'].mean():,.0f}")
    print(f"Total Revenue    : Rs {df['sales'].sum():,.0f}")
    return df

if __name__ == '__main__':
    generate_sales_data()
