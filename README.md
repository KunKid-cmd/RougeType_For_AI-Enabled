# RougeType_For_AI-Enabled
---
This project is developed for the Software Engineering for AI-Enabled Systems course. It integrates a logistic regression model into a Unity-based typing defense game, adapting the difficulty in real time based on player performance.

---
## Project Overview

The system collects gameplay metrics (e.g., typing speed, accuracy, mistake count) and feeds them to an ML model to decide whether to:

- Increase game difficulty

- Keep it constant

- Decrease the difficulty

The model is hosted via a API and responds to Unity's real-time requests.

---

## Directory Structure

- <strong>Model_api_for_Unity.py</strong>: Flask API server that receives player data and returns difficulty predictions.

- <strong>Model_api_for_testing.py</strong>: Utility script for testing the model's API responses.

- <strong>Model_Training.py</strong>: Code for training the logistic regression model using collected player performance data.

- <strong>Model_Versioning_Mlflow.py</strong>: Handles model tracking and version control using MLflow.

- <strong>Model_Explainability&Reasoning.ipynb</strong>: Notebook that explains model predictions using SHAP visualizations.

- <strong>player_performance_log.csv</strong>: Dataset file used to train and validate the model.

- <strong>model.pkl</strong>: Trained logistic regression model file.

- <strong>scaler.pkl</strong>: Scaler used to normalize player performance data.

---
## Setup & Run
Clone the Repository


1. **Clone the Repository**

```bash
git clone https://github.com/KunKid-cmd/RougeType_For_AI-Enabled.git
cd RougeType_For_AI-Enabled
```

2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirement.txt
```

4. **Run the Model API Server**

For API Testing:
```bash
uvicorn Model_api_for_testing:app --reload --port 5050
```

The server will run at `http://127.0.0.1:5050`
Open /docs to interact with the API via Swagger UI.

For API Integration With Unity:

```bash
python Model_api_for_Unity.py
```
This starts a lightweight Flask API for real-time communication with Unity.

---

## Unity Integration

In Unity, use `UnityWebRequest` to send POST requests with this JSON body:

```json
{
  "wpm": 35.0,
  "combo_length": 4,
  "mistake_count": 2,
  "recent_accuracy": 0.88,
  "wave_number": 3
}
```

The server will respond with:

```json
{
  "prediction": "0"
}
```

## Model Training

To retrain the model using updated gameplay data:

```bash
python Model_Training.py
```

The script will:
- Load `player_performance_log.csv`
- Scale the data
- Train a logistic regression model
- Save the model to `model.pkl` and scaler to `scaler.pkl`

---
## Model Explainability and Reasoning
To fulfill:
- **Task 1.4: Model Versioning and Experimentation**
- **Task 1.6: Prediction Reasoning**

We implemented a Jupyter Notebook `Model_Explainability&Reasoning.ipynb` that uses SHAP to explain how each input feature influences the model’s predictions.

###  What It Does

- Loads the trained model and scaler
- Imports a subset of prediction data (player performance)
- Uses SHAP to compute feature impact (e.g., how `wpm`, `accuracy`, `mistake_count`, etc. influence the prediction)
- Visualizes the reasoning

### Example Insight

SHAP analysis confirms that **typing speed** and **accuracy** are the dominant factors in real-time difficulty prediction.

---

## Model Versioning (MLflow)

To enable MLflow tracking:

```bash
python Model_Versioning_Mlflow.py
```

Visit the MLflow UI:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000`.

---

## Example Predictions

| WPM  | Combo | Mistakes | Accuracy | Wave | Prediction   |
|------|-------|----------|----------|-------|--------------|
| 40.2 | 5     | 1        | 0.95     | 2     | 0     |
| 50 | 7     | 1        | 0.9     | 2     | 1     |
| 65 | 18     | 2        | 0.95     | 5     | 2         |

---
