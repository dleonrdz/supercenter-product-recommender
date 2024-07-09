from model_training_and_evaluation_utils import pre_trained_embeddings_faiss_eval
"""
This script is only for call and run the needed steps in order to evaluate and compare
the two main explored approeaches: simple retrieval with embeddings from the pre-trained model
and retrieval with the refined version of the embeddings

The functions are defined on its respective scripts

"""


print('Evaluating two towers architecture embeddings...')
k_values = [2,4,6]
n_values = [8,10]
pre_trained_embeddings_faiss_eval(k_values,n_values)

#print('Evaluating pre-trained embeddings...')
#pre_trained_embeddings_faiss_eval()
#print('Pre trained embeddings evaluation stored')


