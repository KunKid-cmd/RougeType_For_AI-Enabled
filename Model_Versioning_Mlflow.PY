
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load data
df = pd.read_csv("player_performance_log_large.csv")
X = df[["wpm", "combo_length", "mistake_count", "recent_accuracy", "wave_number"]]
y = df["difficulty_label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    # first run
    # model = LogisticRegression(multi_class="multinomial", max_iter=500)

    # second run
    # model = LogisticRegression(multi_class="multinomial", max_iter=1000, C=0.5)

    # third run
    model = LogisticRegression(multi_class="multinomial", max_iter=2000, C=0.3)

    model.fit(X_train, y_train)
    acc = model.score(X_val, y_val)

    # Log to MLflow
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("features", "wpm, combo_length, mistake_count, recent_accuracy, wave_number")

    mlflow.log_metric("validation_accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("MLflow run completed. Accuracy:", acc)
