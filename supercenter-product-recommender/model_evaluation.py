import logging
import os
from recommenders import get_top_n_recommendations_faiss
from model_training_and_evaluation_utils import test_orders_df, leave_k_out, compute_metrics
from db_utilities import write_table
from embedding_process import data_preparation_orders

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_path = os.path.join(PROJECT_ROOT, 'supercenter-product-recommender/logs/model_evaluation_faiss_log.txt')
logging.basicConfig(level=logging.INFO, filename=logs_path, filemode='w')

write_table(test_orders_df, 'test_orders')

logging.info('Hiding test items within orders...')
test_partial_orders_df, test_hidden_orders_df = leave_k_out(test_orders_df, k=2)

logging.info('Preparing test partial orders...')
test_orders_input = data_preparation_orders(test_hidden_orders_df)

logging.info('Getting recommendations...')
text_features = test_orders_input['text_feature'].tolist()
recommendations = get_top_n_recommendations_faiss(text_features, n=5)
test_orders_input['recommendations'] = recommendations

logging.info('Computing metrics...')
precision, recall, f1_score = compute_metrics(test_hidden_orders_df, test_orders_input)

logging.info(f'Precision: {precision:.4f}')
logging.info(f'Recall: {recall:.4f}')
logging.info(f'F1-Score: {f1_score:.4f}')


