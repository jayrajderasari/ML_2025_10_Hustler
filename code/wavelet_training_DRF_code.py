import pandas as pd
import numpy as np
import pandas as pd
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

# --- Read Excel file to DataFrame ---
input_excel_path = "./wavelet_features_reduced.xlsx"
df = pd.read_excel(input_excel_path)

# --- Start H2O and convert to H2OFrame ---
h2o.init()
hf = h2o.H2OFrame(df)
hf['Person'] = hf['Person'].asfactor()

# Initialize DRF model
drf = H2ORandomForestEstimator(
    ntrees=100,
    max_depth=5,
    seed=1
)

# Train the model on the entire dataset
drf.train(y='Person', training_frame=hf)

# Print model performance on the full dataset
performance = drf.model_performance(hf)
print(performance)