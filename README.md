# SuperCenter Product Recommender System

By Ludivine Freymond, Krisztián Kása, Diego León, Alexandra Oliveira & Bendix Sibbel

This repository contains the code for our Capgemini x IE School of Science and Technology final Capstone project. The objective of this project is to enhance the digital transformation strategy of SuperCenter by implementing a robust product recommendation system. This system is designed to improve the customer experience by providing personalized product recommendations based on the items in their shopping cart.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Application Locally](#running-the-application-locally)
- [Disclaimer](#disclaimer)

## Introduction
Our SuperCenter Product Recommender System leverages AI and well-known transformers architectures to offer personalized product recommendations. This project aims to increase the average number of items per shopping cart, enhance cross-selling, and provide tailored offers to customers in online store.

## Project Structure
The repository is organized as follows:

```
.
├── models/
│   └──evaluation_results/
│      └── ... (Evaluation metrics plots for each tested value of k)
├── notebooks/
│   └── ... (Jupyter notebooks used during the EDA and initial model development)
├── streamlit_app/
│   └── app.py (Main Streamlit application)
├── supercenter_product_recommender/
│   └── ... (Core recommendation system scripts and utilities)
├── .gitignore
├── README.md
├── poetry.lock
├── pyproject.toml
└── two_towers_finetuning.py
```
## Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/dleonrdz/supercenter-product-recommender.git
    cd supercenter-product-recommender
    ```

2. **Install dependencies using Poetry**:
    ```sh
    poetry install
    ```

3. **Activate the virtual environment**:
    ```sh
    poetry shell
    ```

4. **Save the challenge raw data**:
    Place the raw data files provided for the challenge within a folder named `data/raw`.

5. **Set up Pinecone API Key**:
    - Obtain your Pinecone API key from the Pinecone console.
    - Create a `.env` file in the root directory and add your Pinecone API key:
        ```
        PINECONE_API_KEY2=your_api_key_here
        ```
   
## Running the Application Locally

To run the application locally, follow these steps:

1. **Replicate the data processing pipeline**:
    Run the following scripts in order:
    - `data_processing.py`
    - `embedding_pipeline.py`
    - `finetuning_trigger.py`
    - `refined_embeddings_storage.py`

    These scripts will process the raw data, create embeddings, fine-tune the two-tower architecture, and store the refined embeddings.

2. **Run the Streamlit app**:
    ```sh
    streamlit run streamlit_app/app.py
    ```

## Evaluation

To replicate our evaluation results, run the `model_evaluation.py` script located in the `supercenter_product_recommender` folder. This script will evaluate the recommendation system's performance and save the results.


## Additional Information

The `two_towers_finetuning.py` script is a copy of the original definition of the two-tower architecture located within the `supercenter_product_recommender` folder. This script is placed in the root directory for the proper working of the Streamlit app.

## Disclaimer

We didn’t deploy the application on Streamlit Cloud for two main reasons:

1. The application reads challenge data that is not meant for public disclosure.
2. GitHub’s storage capabilities were insufficient to store our trained architecture.