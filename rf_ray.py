import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Breast_Cancer.csv')
df.rename(columns={'T Stage ': 'T Stage'}, inplace=True)
df["Grade"] = df["Grade"].apply(lambda x: int(x.replace(" anaplastic; Grade IV", "4")))

# Define categorical columns for one-hot encoding
categorical_cols = ['Race', 'Marital Status', 'A Stage', 'T Stage', 'N Stage',
                     '6th Stage', 'differentiate', 'Estrogen Status', 'Progesterone Status']

# Perform one-hot encoding on categorical columns
onehot_encoder = OneHotEncoder(sparse=False)
encoded_cols = pd.DataFrame(onehot_encoder.fit_transform(df[categorical_cols]))
encoded_cols.columns = onehot_encoder.get_feature_names_out()

# Select numerical columns
numerical_cols = df[['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months', 'Grade']]

# Combine numerical and encoded categorical columns
df_encoded = pd.concat([numerical_cols, encoded_cols, df["Status"]], axis=1)

# Use StandardScaler to normalize numerical features
scaler = StandardScaler()
df_encoded[['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months', 'Grade']] = scaler.fit_transform(
    df_encoded[['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months', 'Grade']])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_encoded.drop("Status", axis=1), df_encoded["Status"], test_size=0.2, random_state=42)

# Define the hyperparameter search space
search_space = {
    "n_estimators": tune.randint(50, 200),
    "max_depth": tune.randint(3, 10),
    "min_samples_split": tune.randint(2, 10),
    "min_samples_leaf": tune.randint(1, 5)
}

num_samples = 20

# Define the training function
def train_model(config):
    # Create the Random Forest classifier based on the hyperparameter configuration
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"]
    )

    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")

    # Report the average F1 score
    f1 = np.mean(scores)
    tune.report(mean_f1_score=f1) 

# Initialize Ray
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

# Get the best hyperparameters and performance
best_config = analysis.get_best_config(metric="mean_f1_score", mode="max")
best_score = analysis.best_result["mean_f1_score"]

# Train the final model using the best hyperparameters
final_model = RandomForestClassifier(
    n_estimators=best_config["n_estimators"],
    max_depth=best_config["max_depth"],
    min_samples_split=best_config["min_samples_split"],
    min_samples_leaf=best_config["min_samples_leaf"]
)
final_model.fit(X_train, y_train)

# Evaluate the final model on the test set
y_pred = final_model.predict(X_test)
test_score = f1_score(y_test, y_pred, average="macro")

# Print the best hyperparameters and performance
print("Best config:", best_config)
print("Best score:", best_score)
print("Test score:", test_score)

# Prepare results dictionary
results = {
    "metric": "f1_macro",
    "best_config": best_config,
    "best_score": best_score,
    "test_score": test_score,
    "grid_tried": list(analysis.get_all_configs()),
    "trial_dataframes": [df.to_dict(orient="list") for df in analysis.trial_dataframes.values()],
    "model": str(final_model),
    "num_samples": num_samples
}

# Create results directory if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

# Save results to a JSON file
with open("results/random_forest_ray.json", "w") as f:
    json.dump(results, f, indent=4)

# Calculate and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Save the confusion matrix to a text file
with open("results/random_forest_ray_confusion_matrix.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm, separator=", "))

# Shutdown Ray
ray.shutdown()
