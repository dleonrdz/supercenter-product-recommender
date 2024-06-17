import pandas as pd
import random
from db_utilities import read_table
import numpy as np

orders_df = read_table('processed_orders_data')
orders_df['order_id'] = orders_df['order_id'].astype(str)
test_size = 0.3
test_orders = orders_df['order_id'].drop_duplicates().sample(frac=test_size, random_state=42)

test_orders_df = orders_df[orders_df['order_id'].isin(test_orders)]
train_orders_df = orders_df[(~orders_df['order_id'].isin(test_orders))]
def leave_k_out_efficient(df, k=1):
    # Randomly permute the indices within each group
    df['rand'] = np.random.random(len(df))
    df = df.sort_values(['order_id', 'rand'])
    # Generate a rank within each group
    df['rank'] = df.groupby('order_id').cumcount() + 1

    # Determine the number of items in each order
    order_sizes = df.groupby('order_id')['product_id'].transform('count')

    # Create masks for partial orders and hidden items
    df['is_hidden'] = df['rank'] <= k
    partial_orders_df = df[~df['is_hidden']].drop(columns=['rand', 'rank', 'is_hidden'])
    hidden_items_df = df[df['is_hidden']].drop(columns=['rand', 'rank', 'is_hidden'])

    return partial_orders_df, hidden_items_df
