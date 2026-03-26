"""
Pipeline Model Deployment
"""
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import os
import pandas as pd
from prefect import flow, task, get_run_logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score




@task(description="Loading the data", retries=5, retry_delay_seconds=10)
def load_data(path: str) -> pd.DataFrame:
    """
    Load data using pandas parquet and just return the first 1000 record becuase of memeory
    """
    logger = get_run_logger()
    if not os.path.exists(path):
        logger.error(f"The File path {path} does not exist")
        raise FileNotFoundError(f"Missing File: {path}")
    try:
        df = pd.read_parquet(path).head(1000)
        if df.empty:
            logger.error("The parquet file is empty")
            raise ValueError("empty file")
    except Exception as e:
        logger.error(f"Faild to load parqut file due to: {e}")
        raise e
    logger.info(f"Successfully Loaded the dataset {df.shape}")
    return df


@task(description="Processing data")
def process_data(df: pd.DataFrame) -> tuple:
    """
    Clean and exctract usfull data for the model
    """
    logger = get_run_logger()

    required_columns = ['PULocationID', 'DOLocationID', 'trip_distance', 'lpep_dropoff_datetime', 'lpep_pickup_datetime']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Data is missing required column: {col}")
            raise KeyError(f"Missing column: {col}")
        
    try:
        df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
        df_encoded = pd.get_dummies(df[['PU_DO', 'trip_distance']], columns=['PU_DO'])
        df_encoded["duration"] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
        df_encoded.duration = df_encoded.duration.apply(lambda td: td.total_seconds() / 60)
        Y = df_encoded.pop("duration")
    except Exception as e:
        logger.error(f"Processing data error due to {e}")
        raise e
    logger.info("Processing data and feature engineering and encoding complete.")
    return df_encoded, Y


@task(description="training data")
def firstTrain(X_train: pd.DataFrame, Y_train: np.ndarray) -> None:
    """
    Start the first stage of training to exctract the best parameter
    """
    
    logger = get_run_logger()

    def hyperTrain(params: dict) -> dict:
        """
        Log and Train using hyperopt so you can choose the best models
        """
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model", "RandomForset")
            mlflow.log_params(params)

            model = RandomForestRegressor(**params, random_state=42)

            # Perform 5-Fold Cross Validation. 
            # n_jobs=-1 tells it to use all your CPU cores to speed it up.
            # Scikit-learn uses negative values for errors, so we ask for 'neg_root_mean_squared_error'
            score = cross_val_score(
                model, X_train, Y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
            )
            rmse = abs(score.mean())

            # y_pred = model.predict(X_val)
            # rmse = root_mean_squared_error(y_pred=y_pred, y_true=Y_val)

            mlflow.log_metrics({"rmse": rmse})

        return {"loss": rmse, "status": STATUS_OK}
    

    space = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 200, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 10, 40, 1)),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 5, 1)),
        "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0, 0.5),
        "bootstrap": hp.choice("bootstrap", [True, False]),
    }

    rstate = np.random.default_rng(42)
    trails = Trials()
    try:
        with mlflow.start_run(run_name="HyperOpt_Optmization"):
            fmin(
                fn=hyperTrain,
                space=space,
                algo=tpe.suggest,
                max_evals=15,
                trials=trails,
                rstate=rstate,
            )
            currently_id = mlflow.active_run().info.run_id
        logger.info(f"Successfully trained the model with run id of {currently_id}")
    except Exception as e:
        logger.error(f"Faild to train due to {e}")
        raise e


@task(description="evaluating models")
def evaluation(name: str, x_test: pd.DataFrame, y_test: np.ndarray, x_train: pd.DataFrame, y_train: np.ndarray, client):
    """
    Evaluate the top 5 models and choose the best one 
    """
    logger = get_run_logger()

    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        logger.error(f"There is no Experiment by the name {name}")
        raise ValueError("Wrnog Name")

    try:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by = ["metrics.rmse ASC"]
        )

        if len(runs) == 0:
            logger.error(f"No Runs founds in Experiment {experiment}")
            raise ValueError(f"No Runs Founds in Experiment {experiment}")

        # Ensure we don't accidentally grab the "Parent" run by filtering out runs without the metric
        valid_runs = [r for r in runs if "rmse" in r.data.metrics]

        top_5_runs = valid_runs[:5]
        best_test_rmse = float('inf')
        ultimate_best_model = None
        ultimate_best_params = None

        run_name = f"Final_Eval_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
        with mlflow.start_run(run_name=run_name):
            for run in top_5_runs:
                params = run.data.params
                if len(params) == 0:
                    logger.error(f"No Parameters Found in run: {run}")
                    raise ValueError(f"No Parameters Found in run: {run}")

                # Convert string params back to proper types
                n_estimators = int(float(params['n_estimators']))
                max_depth = int(float(params['max_depth']))
                min_samples_split = int(float(params['min_samples_split']))
                min_weight_fraction_leaf = float(params['min_weight_fraction_leaf'])
                bootstrap = params['bootstrap'].lower() == 'true'

                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, 
                    min_samples_split=min_samples_split, 
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    bootstrap=bootstrap, random_state=42
                )
                model.fit(x_train, y_train)

                
                # Evaluate on the unseen test data
                y_pred = model.predict(x_test)
                test_rmse = root_mean_squared_error(y_test, y_pred)
                
                # cv_rmse = run.data.metrics['cv_mean_rmse']
                # print(f"Run {run_id[:8]} | CV RMSE: {cv_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

                # Track the absolute best performer on the test set
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    ultimate_best_model = model
                    ultimate_best_params = params

            mlflow.log_params(ultimate_best_params)
            mlflow.log_metric("final_test_rmse", best_test_rmse)
            mlflow.sklearn.log_model(ultimate_best_model, artifact_path="best_rf_model")

            best_run_id = mlflow.active_run().info.run_id

        deleted_count = 0
        
        for run in valid_runs:
            if run.info.run_id != best_run_id:
                client.delete_run(run.info.run_id)
                deleted_count += 1

        parent_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'HyperOpt_Optmization'"
        )
        for p_run in parent_runs:
            client.delete_run(p_run.info.run_id)

        logger.info("Successfully Evaluated the models and deleted the others")
    except Exception as e:
        logger.error(f"The Evaluation faild due to {e}")
        raise e    
    print(f"Evaluating model Succfully model_id {best_run_id}.")
    return best_run_id


@task(description="register Model", retries=5, retry_delay_seconds=10)
def registerModel(client, model_run_id, model_name="best_model") -> None:
    """
    Register the Choosen Model and return it's parameter
    """
    logger = get_run_logger()
    if not model_run_id:
        logger.error(f"There is No model id {model_run_id}")
        raise ValueError("No model id")
    model_uri = f"runs:/{model_run_id}/best_rf_model"
    try:
        register_model = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Optional but recommended: Tag it as your production model using an alias
        client.set_registered_model_alias(
            name=model_name, 
            alias="production", 
            version=register_model.version
        )
        logger.info(f"Successfully Register the model {model_name}")
    except Exception as e:
        logger.error(f"Faild to register model due to {e}")
        raise e
