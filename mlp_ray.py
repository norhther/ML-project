import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import ray
from ray import tune
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import json
import os

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Breast_Cancer.csv')
df.rename(columns={'T Stage ': 'T Stage'}, inplace=True)
df["Grade"] = df["Grade"].apply(lambda x: int(x.replace(" anaplastic; Grade IV", "4")))

categorical_cols = ['Race', 'Marital Status', 'A Stage', 'T Stage', 'N Stage',
                     '6th Stage', 'differentiate', 'Estrogen Status', 'Progesterone Status']

onehot_encoder = OneHotEncoder(sparse=False)
encoded_cols = pd.DataFrame(onehot_encoder.fit_transform(df[categorical_cols]))
encoded_cols.columns = onehot_encoder.get_feature_names_out()

numerical_cols = df[['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months', 'Grade']]

df_encoded = pd.concat([numerical_cols, encoded_cols, df["Status"]], axis=1)

scaler = StandardScaler()
df_encoded[['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months', 'Grade']] = scaler.fit_transform(
    df_encoded[['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months', 'Grade']])

X_train, X_test, y_train, y_test = train_test_split(df_encoded.drop("Status", axis=1), df_encoded["Status"], test_size=0.2, random_state=42)

# Define the hyperparameter search space
search_space = {
    "hidden_layer_sizes": tune.choice([(64,), (128,), (256,), (64, 64), (128, 128), (256, 256)]),
    "activation": tune.choice(["relu", "tanh"]),
    "alpha": tune.loguniform(1e-5, 1e-1)
}

num_samples = 20

# Define the training function
def train_model(config):
    model = MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation=config["activation"],
        alpha=config["alpha"],
        max_iter=1000
    )

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
    f1 = np.mean(scores)
    tune.report(mean_f1_score=f1) 

ray.init()

# Run the hyperparameter search with cross-validation
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=num_samples,
    resources_per_trial={"cpu": 8},
    metric="mean_f1_score",
    mode="max",
    name="train_model"
)

best_config = analysis.get_best_config(metric="mean_f1_score", mode="max")
best_score = analysis.best_result["mean_f1_score"]

final_model = MLPClassifier(
    hidden_layer_sizes=best_config["hidden_layer_sizes"],
    activation=best_config["activation"],
    alpha=best_config["alpha"]
)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
test_score = f1_score(y_test, y_pred, average="macro")

print("Best config:", best_config)
print("Best score:", best_score)
print("Test score:", test_score)

trial_dataframes = [df.to_dict(orient="list") for df in analysis.trial_dataframes.values()]

results = {
    "metric": "f1_macro",
    "best_config": best_config,
    "best_score": best_score,
    "test_score": test_score,
    "grid_tried": list(analysis.get_all_configs()),
    "trial_dataframes": trial_dataframes,
    "model": str(final_model),
    "num_samples": num_samples
}

if not os.path.exists("results"):
    os.makedirs("results")
with open("results/mlp_ray.json", "w") as f:
    json.dump(results, f, indent=4)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

with open("results/mlp_ray_confusion_matrix.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm, separator=", "))

ray.shutdown()
