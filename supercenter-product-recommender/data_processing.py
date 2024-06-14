from data_processing_utilities import raw_data_processing
from db_utilities import write_table

# Processing raw data
print('Processing raw data...')
orders_data, products_data = raw_data_processing()

# Saving processed data
print('Saving processed data...')
write_table(orders_data, 'processed_orders_data')
write_table(products_data, 'processed_products_data')

print('Processed data uploaded')
