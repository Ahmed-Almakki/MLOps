from io import BytesIO
import os
import sys
import pandas as pd
import pickle
from pathlib import Path
import requests


BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "model" / "model.bin", "rb") as f:
    dv, model = pickle.load(f)


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    if s3_endpoint_url is not None and filename.startswith('s3://'):
        print("Reading data from S3-compatible storage...")
        options = {'client_kwargs': {'endpoint_url': s3_endpoint_url}}
        return pd.read_parquet(filename, storage_options=options)
    print("Reading data from URL...")
    response = requests.get(filename)
    response.raise_for_status()
    df = pd.read_parquet(BytesIO(response.content))
    return df

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def save_data(df, output_file):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    options = {'client_kwargs': {'endpoint_url': s3_endpoint_url}}
    df.to_parquet(output_file, storage_options=options, index=False)


def main(year, month):
    categorical = ['PULocationID', 'DOLocationID']
    try:
        input_file = get_input_path(year, month)
        output_file = get_output_path(year, month)

        print(f"Reading data from {input_file}...")
        df = read_data(input_file)

        print("Preparing data...")
        df = prepare_data(df, categorical)

        print("Predicting durations...")
        dicts = df[categorical].to_dict(orient='records')

        print("Saving results...")
        X_val = dv.transform(dicts)

        print("Making predictions...")
        y_pred = model.predict(X_val)

        df_result = df.copy()
        df_result['predicted_duration'] = y_pred
        save_data(df_result, output_file)

        return y_pred.mean()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    print("Calculating the average predicted duration for the given month and year...")
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    print(f"Processing data for year: {year}, month: {month}...")
    result = main(year, month)
    print(f"Average predicted duration: {result:.2f} minutes")
