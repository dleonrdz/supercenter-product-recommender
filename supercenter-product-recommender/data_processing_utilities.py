import pandas as pd
import os

# Calculate the path to the directory ABOVE the current script directory (i.e., the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(PROJECT_ROOT, "data/raw/")
def raw_data_processing():
    '''
    Ths function takes raw data and prepares the data for further modeling and analyses by merging all raw order files
    with product, departments and aisles information
    '''

    # Read all orders tables and append them into one
    orders_df = pd.DataFrame()
    for i in range(1, 6):
        path = f'{raw_data_path}tabla_ordenes_{str(i)}.xlsx'
        orders = pd.read_excel(path)
        orders_df = orders_df._append(orders, ignore_index=True)

    # Read the rest of information
    products_df = pd.read_excel(f'{raw_data_path}/tabla_producto.xlsx')
    departments_df = pd.read_excel(f'{raw_data_path}/tabla_departamento.xlsx')
    aisles_df = pd.read_excel(f'{raw_data_path}/tabla_pasillos.xlsx')

    orders_df['id_producto'] = orders_df['id_producto'].astype(str)
    products_df['id_producto'] = products_df['id_producto'].astype(str)
    products_df['id_departamento'] = products_df['id_departamento'].astype(str)
    products_df['id_pasillo'] = products_df['id_pasillo'].astype(str)
    departments_df['id_departamento'] = departments_df['id_departamento'].astype(str)
    aisles_df['id_pasillo'] = aisles_df['id_pasillo'].astype(str)

    # First step is to join products with department and aisle
    products_df2 = products_df \
        .merge(departments_df,
               on='id_departamento',
               how='left') \
        .merge(aisles_df,
               on='id_pasillo',
               how='left')

    # The next step is to join the orders with the products including department and aisle information
    df = orders_df.merge(products_df2,
                         on='id_producto',
                         how='left')

    df.fillna('Missing',inplace=True)

    # Rename columns
    renames= {'id_linea':'row_id',
              'id_orden':'order_id',
              'id_producto':'product_id',
              'incluido_orden_carrito':'cart_inclusion_order',
              'reordenado':'reordered',
              'nombre_producto':'product_name',
              'id_pasillo':'aisle_id',
              'id_departamento':'department_id',
              'departamento':'department',
              'pasillo':'aisle'}

    df.rename(columns=renames, inplace=True)
    return df



