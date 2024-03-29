# SPDX-License-Identifier: MIT
# (c) 2019 Jim Thompson


import mlflow
import os
from random import random, randint

if __name__ == "__main__":
    print("Running mlflow_tracking.py")

    print(mlflow.__version__)

    print(mlflow.get_tracking_uri())

    experiment_id = mlflow.set_experiment('hyperparms')

    with mlflow.start_run() as run:
        mlflow.log_param("param1", randint(0, 100))

        mlflow.log_metric("foo", random())
        mlflow.log_metric("foo2", random() + 1)
        mlflow.log_metric("foo3", random() + 2)

        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        with open("outputs/test.txt", "w") as f:
            f.write("hello world!")

        mlflow.log_artifacts("outputs")



