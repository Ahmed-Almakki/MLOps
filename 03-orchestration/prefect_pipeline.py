"""
Pipeline Model Deployment
"""
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

EXPERIMENT_NAME = "RandomForestPrefect"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()


def load_data(path: str) -> pd.DataFrame:
    """
    Load data using pandas parquet and just return the first 1000 record becuase of memeory
    """
    df = pd.read_parquet(path).head(1000)
    return df


def process_data(df: pd.DataFrame) -> tuple:
    """
    Clean and exctract usfull data for the model
    """
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    df_encoded = pd.get_dummies(df[['PU_DO', 'trip_distance']], columns=['PU_DO'])
    df_encoded["duration"] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df_encoded.duration = df_encoded.duration.apply(lambda td: td.total_seconds() / 60)
    Y = df_encoded.pop("duration")
    return df_encoded, Y



def firstTrain(X_train: pd.DataFrame, Y_train: np.ndarray) -> None:
    """
    Start the first stage of training to exctract the best parameter
    """

    def hyperTrain(params: dict) -> dict:
        """
        Log and Train using hyperopt so you can choose the best models
        """
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model", "RandomForset")
            mlflow.sklearn.autolog()

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

    with mlflow.start_run(run_name="HyperOpt_Optmization"):
        fmin(
            fn=hyperTrain,
            space=space,
            algo=tpe.suggest,
            max_evals=15,
            trials=trails,
            rstate=rstate,
        )

def evaluation(name: str, x_test: pd.DataFrame, y_test: np.ndarray, x_train: pd.DataFrame, y_train: np.ndarray):
    """
    Evaluate the top 5 models and choose the best one 
    """
    experiment = client.get_experiment_by_name(name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by = ["metrics.rmse ASC"]
    )

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
    
    return best_run_id


def registerModel(model_run_id, model_name="best_model") -> None:
    """
    Register the Choosen Model and return it's parameter
    """
    model_uri = f"runs:/{model_run_id}/best_rf_model"

    register_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Optional but recommended: Tag it as your production model using an alias
    client.set_registered_model_alias(
        name=model_name, 
        alias="production", 
        version=register_model.version
    )


def main_pipeline():
    file_path = "../02-experiment-tracking/data/green_tripdata_2023-01.parquet"
    df = load_data(file_path)
    print("data loaded successfully")
    x, y = process_data(df)
    print("data have been processed succussfully")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    firstTrain(x_train, y_train)
    print(f"best params")
    model_id = evaluation(EXPERIMENT_NAME, x_test, y_test, x_train, y_train)
    registerModel(model_id)

    

main_pipeline()