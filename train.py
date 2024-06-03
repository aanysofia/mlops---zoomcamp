import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)



@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    # Set the experiment name
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("taxi_duration_prediction")



    # Load the training and validation datasets
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))


    # Convert sparse matrix to dense numpy array
    X_train = X_train.toarray()
    X_val = X_val.toarray()

    # # Load the training and validation datasets
    # X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    # X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    # Start an MLflow run
    with mlflow.start_run():
        # Initialize and train the RandomForestRegressor model
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = rf.predict(X_val)

        # Calculate the RMSE
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f'RMSE: {rmse}')

        # Log the RMSE
        mlflow.log_metric("rmse", rmse)

if __name__ == '__main__':
    run_train()
