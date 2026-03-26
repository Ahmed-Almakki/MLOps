from prefect import flow
from prefect_pipeline import *

EXPERIMENT_NAME = "RandomForestPrefect_v2"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")



@flow(name="NYC Taxi Orchestrator")
def main():
    logger = get_run_logger()
    file_path = "../02-experiment-tracking/data/green_tripdata_2023-01.parquet"
    
    logger.info("Starting Pipeline...")
    df = load_data(file_path)
    
    logger.info("Processing Data...")
    x, y = process_data(df)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    logger.info("Executing Hyperopt Training...")
    firstTrain(x_train, y_train)
    
    logger.info("Evaluating and finding best model...")
    model_id = evaluation(EXPERIMENT_NAME, x_test, y_test, x_train, y_train, client)
    
    logger.info("Registering final model...")
    registerModel(client, model_id)
    
    logger.info("Pipeline Complete!")
    

if __name__ == "__main__":
    main.serve(
        name="taxi_training",
        tags=["ML"],
        cron="*/3 * * * *"
    )