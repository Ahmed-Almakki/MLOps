from datetime import datetime
import os
import pandas as pd


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

options = {
    'client_kwargs': {
        'endpoint_url': 'http://localhost:4566'
    }
}

input_file = 's3://nyc-duration/in/2023-01.parquet'

# save file inside the container to S3-compatible storage
df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

os.system('python batch.py 2023 1')
output_file = 's3://nyc-duration/out/2023-01.parquet'
df_output = pd.read_parquet(output_file, storage_options=options)
print("df_output:", df_output.info())

print(f"Sum of predicted durations: {df_output['predicted_duration'].sum():.2f} minutes")