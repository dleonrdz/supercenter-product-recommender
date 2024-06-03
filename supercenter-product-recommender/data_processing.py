from data_processing_utilities import raw_data_processing
from db_utilities import write_table

# Processing raw data
print('Processing raw data...')
processed_data = raw_data_processing()

# Saving processed data
print('Saving processed data...')
write_table(processed_data, 'processed_orders_data')

print('Process data uploaded')
