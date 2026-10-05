import os

# Keep test logs free of MLflow's interactive hints.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
