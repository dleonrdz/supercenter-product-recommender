import tensorflow as tf
from two_towers_finetuning import TwoTowerModel
print('Loading model...')
# Model needs to be read at first
model = tf.keras.models.load_model('models/two_tower_architecture', custom_objects={'TwoTowerModel': TwoTowerModel})
print('Model loaded')
import pandas as pd
from recommenders import get_top_n_recommendations_faiss, get_top_n_recommendations_pinecone_batch
from db_utilities import write_table, read_table, update_table
from embedding_process import data_preparation_orders_test

"""
This script defines the functions for models evaluation.
"""

# Reading test orders
test_orders_df = read_table('test_orders')
test_orders_df['order_id'] = test_orders_df['order_id'].astype(str)
test_order_ids = list(test_orders_df['order_id'].unique())

# Apply pre-processing to test orders
test_orders_df = data_preparation_orders_test(test_orders_df)
def leave_k_out(df, k=1):
    """
    This function applies the leave-k-out methodology to the given dataframe.
    It consists on hiding the last k added products of each order, based on the
    cart inclusion order
    """

    # Sort df by order and inclusion order of the products
    df = df.sort_values(['order_id', 'cart_inclusion_order'])

    # Create the hidden flag to the last k products by order
    df['is_hidden'] = df.groupby('order_id')['cart_inclusion_order'].transform(
        lambda x: x.rank(method='first', ascending=False) <= k)

    # Defining the df without hidden products
    partial_orders_df = df[~df['is_hidden']].drop(columns=['is_hidden'])

    # Defining a separated df with the hidden products
    hidden_items_df = df[df['is_hidden']].drop(columns=['is_hidden'])

    return partial_orders_df, hidden_items_df

def compute_metrics(test_hidden_orders_df, test_partial_orders_df):
    """
    This function computes the recall, precision and recall, given the dataframe with
    partial order and the respective recommendations, and the hidden products of each
    of the orders
    """

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

def pre_trained_embeddings_faiss_eval(k_values, n_values, algorithm='pre-trained embeddings'):
    """
    This function applies the leave-k-out methodology and metrics computation with the simple
    recommendations retrieval by using the pre-trained model embeddings. It is defined so that
    the user can try different values of k and n (recommendations)
    The results are saved within the 'models_evaluation' table in the data/processed_data.db database
    """
    # Initialize an empty list to collect the rows for the DataFrame
    rows = []
    for k in k_values:
        for n in n_values:
            print(f'Evaluating k={k} and n={n}')
            test_partial_orders_df, test_hidden_orders_df = leave_k_out(test_orders_df, k=k)
            test_partial_orders_df = test_partial_orders_df.sort_values(['order_id', 'cart_inclusion_order'])
            test_orders_input = test_partial_orders_df

            test_orders_input = test_orders_input.loc[test_orders_input.groupby('order_id')['cart_inclusion_order'].idxmax()]

            text_features = test_orders_input['order_text_feature_val'].tolist()

            recommendations = get_top_n_recommendations_faiss(text_features, n=n)
            test_orders_input['recommendations'] = recommendations

            precision, recall, f1_score = compute_metrics(test_hidden_orders_df, test_orders_input)

            # Append the metrics to the rows list
            rows.append({'k': k, 'n': n, 'metric': 'precision', 'value': precision, 'algorithm': algorithm})
            rows.append({'k': k, 'n': n, 'metric': 'recall', 'value': recall, 'algorithm': algorithm})
            rows.append({'k': k, 'n': n, 'metric': 'f1', 'value': f1_score, 'algorithm': algorithm})

            print(f'k={k}, n={n}')
            print(f'Precision={precision}')
            print(f'Recall={recall}')
            print(f'F1={f1_score}')

        print(f'Evaluation for k={k}, n={n} is complete and stored')
    eval_df = pd.DataFrame(rows)
    write_table(eval_df, 'models_evaluation')
    print(f'Evaluation for k values = {k_values} and n values = {n_values}')

def two_towers_embeddings_eval(k_values, n_values, algorithm='two-tower architecture'):
    """
        This function applies the leave-k-out methodology and metrics computation with the simple
        recommendations retrieval by using the pre-trained model embeddings. It is defined so that
        the user can try different values of k and n (recommendations)
        The results are saved within the 'models_evaluation' table in the data/processed_data.db database
        """
    # Initialize an empty list to collect the rows for the DataFrame
    rows = []
    for k in k_values:
        for n in n_values:
            test_partial_orders_df, test_hidden_orders_df = leave_k_out(test_orders_df, k=k)
            test_orders_input = data_preparation_orders_test(test_hidden_orders_df)
            test_orders_input = test_orders_input.loc[
                test_orders_input.groupby('order_id')['cart_inclusion_order'].idxmax()]

            text_features = test_orders_input['order_text_feature'].tolist()
            recommendations = get_top_n_recommendations_pinecone_batch('supercenter-recommender-system',
                                                                       text_features,
                                                                       model,
                                                                       n=n)
            test_orders_input['recommendations'] = recommendations

            precision, recall, f1_score = compute_metrics(test_hidden_orders_df, test_orders_input)

            # Append the metrics to the rows list
            rows.append({'k': k, 'n': n, 'metric': 'precision', 'value': precision, 'algorithm': algorithm})
            rows.append({'k': k, 'n': n, 'metric': 'recall', 'value': recall, 'algorithm': algorithm})
            rows.append({'k': k, 'n': n, 'metric': 'f1', 'value': f1_score, 'algorithm': algorithm})

            print(f'k={k}, n={n}')
            print(f'Precision={precision}')
            print(f'Recall={recall}')
            print(f'F1={f1_score}')

        print(f'Evaluation for k={k}, n={n} is complete and stored')
    eval_df = pd.DataFrame(rows)
    update_table(eval_df, 'models_evaluation')
    print(f'Evaluation for k values = {k_values} and n values = {n_values}')

