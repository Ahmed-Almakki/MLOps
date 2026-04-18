import mlflow

# Point to your server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

run_id = "e4627f5d28ba473183fb9f307316446f"

# This downloads the artifacts and returns the local path
local_path = mlflow.artifacts.download_artifacts(run_id=run_id)

print(f"Model downloaded to: {local_path}")