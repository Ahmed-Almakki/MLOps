import mlflow
from prefect import task, flow
import sklearn
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet('./data/yellow_tripdata_2023-03.parquet')
    