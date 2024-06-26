import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Calculate the path to the directory ABOVE the current script directory (i.e., the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the absolute path to the database, assuming the database is in the project root
db_path = os.path.join(PROJECT_ROOT, "data/processed_data.db")
db_url = f"sqlite:///{db_path}"

engine = create_engine(db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def write_table(df: pd.DataFrame, table: str):
    """
    Writes the specified DataFrame to the specified table in store_sales.db
    """
    with SessionLocal() as session:
        df.to_sql(table, session.get_bind(), if_exists="replace", index=False)
def read_table(table: str):
    """
    Reads the specified table from store_sales.db and returns the result as a DataFrame
    """
    query = f"SELECT * FROM {table}"
    with SessionLocal() as session:
        q = text(query)
        df = pd.DataFrame(session.execute(q))

    return df

def delete_table(table: str):
    """
    Deletes the specified table from store_sales.db and returns the result as a DataFrame
    """
    query = f"SELECT * FROM {table}"
    with SessionLocal() as session:
        q = text(query)
        response = session.execute(q)

    return response

