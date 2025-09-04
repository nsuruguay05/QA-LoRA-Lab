from TrainQA import TrainQA
import pandas as pd
from itertools import product


if __name__ == "__main__":
    # Hyperparameters to use in hyperparameter search
    hyperparams = {
        "r": [4, 8, 16, 32],
        "batch_size": [1, 2, 4, 8]
    }

    # Generate all combinations of hyperparameters
    hyperparam_combinations = list(product(*hyperparams.values()))

    # Generate a unique identifier for each run
    base_folder = "lora/Qwen2.5-0.5B-Instruct/{timestamp}/"
    new_model_name = "Qwen2.5-0.5B-Instruct-lora-{timestamp}"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Initialize the training class with the model and dataset paths
    train_qa = TrainQA(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        new_model_name=new_model_name.format(timestamp=timestamp),
        base_folder=base_folder.format(timestamp=timestamp),
        train_dataset="dataset_covid_qa_train.json",
        val_dataset="dataset_covid_qa_dev_gold.json",
        qlora=False
    )

    # Iterate over all hyperparameter combinations
    for r, batch_size in hyperparam_combinations:
        # Train the model
        train_qa.train(r=r, batch_size=batch_size, epochs=2, eval_and_profile=True)

        # Save metrics to a CSV file with identifier and training hyperparameters
        hyperparams = {
            "model_name": train_qa.model_name,
            "qlora": False,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "r": r,
            "epochs": 2,
            "batch_size": batch_size,
            "learning_rate": 2e-4
        }
        metrics = train_qa.metrics
        df = pd.DataFrame([{
            "timestamp": timestamp,
            **hyperparams,
            **metrics
        }])
        df.to_csv("training_metrics.csv", mode='a', header=not pd.io.common.file_exists("training_metrics.csv"), index=False)

        # Reset the training class for the next run
        train_qa.reset()
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        train_qa.new_model_name = new_model_name.format(timestamp=timestamp)
        train_qa.base_folder = base_folder.format(timestamp=timestamp)
