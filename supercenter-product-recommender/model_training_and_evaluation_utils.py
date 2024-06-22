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

def leave_k_out(df, k=1):
    df = df.sort_values(['order_id', 'cart_inclusion_order'])
    df['is_hidden'] = df.groupby('order_id')['cart_inclusion_order'].transform(
        lambda x: x.rank(method='first', ascending=False) <= k)

    partial_orders_df = df[~df['is_hidden']].drop(columns=['is_hidden'])
    hidden_items_df = df[df['is_hidden']].drop(columns=['is_hidden'])

    return partial_orders_df, hidden_items_df

def compute_metrics(test_hidden_orders_df, test_partial_orders_df):
    # Explode the DataFrame to have one row per actual item
    actual_items_df = test_hidden_orders_df[['order_id', 'product_id']].copy()
    actual_items_df['is_actual'] = 1

    # Explode the DataFrame to have one row per recommended item
    recommendations_df = test_partial_orders_df[['order_id', 'recommendations']].copy()
    recommendations_df = recommendations_df.explode('recommendations')
    recommendations_df.rename(columns={'recommendations': 'product_id'}, inplace=True)
    recommendations_df['is_recommended'] = 1

    # Merge the actual items with the recommended items
    merged_df = pd.merge(recommendations_df, actual_items_df, how='left', on=['order_id', 'product_id'])

    # Calculate true positives, false positives, and false negatives
    true_positive = merged_df['is_actual'].sum()
    false_positive = len(merged_df) - true_positive
    false_negative = actual_items_df.shape[0] - true_positive

    # Calculate precision, recall, and F1-score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score