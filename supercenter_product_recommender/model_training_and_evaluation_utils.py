
import pandas as pd
import tensorflow as tf
import os
from recommenders import get_top_n_recommendations_faiss, get_top_n_recommendations_pinecone_batch
from db_utilities import write_table, read_table
from embedding_process import data_preparation_orders
from two_towers_finetuning import TwoTowerModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
evaluation_storage_path = os.path.join(PROJECT_ROOT, 'data/results/')
with tf.keras.utils.custom_object_scope({'TwoTowerModel': TwoTowerModel}):
    model = tf.keras.models.load_model('models/two_towers_trained.keras')

test_orders_df = read_table('test_orders')
test_orders_df['order_id'] = test_orders_df['order_id'].astype(str)
test_order_ids = list(test_orders_df['order_id'].unique())


k_values = [2,4,6]
n_values = [3,5,10]
precision_df1 = pd.DataFrame(index=k_values, columns=n_values)
recall_df1 = pd.DataFrame(index=k_values, columns=n_values)
f1_df1 = pd.DataFrame(index=k_values, columns=n_values)

precision_df2 = pd.DataFrame(index=k_values, columns=n_values)
recall_df2 = pd.DataFrame(index=k_values, columns=n_values)
f1_df2 = pd.DataFrame(index=k_values, columns=n_values)

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

def pre_trained_embeddings_faiss_eval():
    for k in k_values:
        for n in n_values:
            print(f'Evaluating k={k} and n={n}')
            test_partial_orders_df, test_hidden_orders_df = leave_k_out(test_orders_df, k=k)
            test_orders_input = data_preparation_orders(test_hidden_orders_df)

            text_features = test_orders_input['text_feature'].tolist()
            recommendations = get_top_n_recommendations_faiss(text_features, n=n)
            test_orders_input['recommendations'] = recommendations

            precision, recall, f1_score = compute_metrics(test_hidden_orders_df, test_orders_input)

            precision_df1.loc[k, n] = precision
            recall_df1.loc[k, n] = recall
            f1_df1.loc[k, n] = f1_score

            print(f'k={k}, n={n}')
            print(f'Precision={precision}')
            print(f'Recall={recall}')
            print(f'F1={f1_score}')

        print(f'Evaluation for k={k}, n={n} is complete and stored')

    precision_df1.to_csv(os.path.join(evaluation_storage_path, 'precision_pt1.csv'))
    recall_df1.to_csv(os.path.join(evaluation_storage_path, 'recall_pt1.csv'))
    f1_df1.to_csv(os.path.join(evaluation_storage_path, 'f1_pt1.csv'))

def two_towers_embeddings_eval():
    for k in k_values:
        for n in n_values:
            test_partial_orders_df, test_hidden_orders_df = leave_k_out(test_orders_df, k=k)
            test_orders_input = data_preparation_orders(test_hidden_orders_df)

            text_features = test_orders_input['text_feature'].tolist()
            recommendations = get_top_n_recommendations_pinecone_batch('supercenter-recommender-system',
                                                                       text_features,
                                                                       n=n)
            test_orders_input['recommendations'] = recommendations

            precision, recall, f1_score = compute_metrics(test_hidden_orders_df, test_orders_input)

            precision_df1.loc[k, n] = precision
            recall_df1.loc[k, n] = recall
            f1_df1.loc[k, n] = f1_score

            print(f'k={k}, n={n}')
            print(f'Precision={precision}')
            print(f'Recall={recall}')
            print(f'F1={f1_score}')

    precision_df2.to_csv(os.path.join(evaluation_storage_path, 'precision_pt2.csv'))
    recall_df2.to_csv(os.path.join(evaluation_storage_path, 'recall_pt2.csv'))
    f1_df2.to_csv(os.path.join(evaluation_storage_path, 'f1_pt2.csv'))
