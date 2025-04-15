import numpy as np
import pandas as pd
import h2o
from h2o.estimators.xgboost import H2OXGBoostEstimator

# --- Read Excel file to DataFrame ---
input_excel_path = "./wavelet_features_reduced.xlsx"
df = pd.read_excel(input_excel_path)

# --- Start H2O and convert to H2OFrame ---
h2o.init()
hf = h2o.H2OFrame(df)
hf['Person'] = hf['Person'].asfactor()

# Initialize XGBoost model
xgb = H2OXGBoostEstimator(
    ntrees=100,
    max_depth=5,
    learn_rate=0.1,
    seed=1
)
print("start training")

# Train the model on the entire dataset
xgb.train(y='Person', training_frame=hf)

# Print model performance on the full dataset
performance = xgb.model_performance(hf)
print(performance)
