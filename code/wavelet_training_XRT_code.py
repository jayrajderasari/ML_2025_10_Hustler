import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

print('start program')

# --- Read Excel file to DataFrame ---
input_excel_path = "./wavelet_features_reduced.xlsx"
df = pd.read_excel(input_excel_path)

print(f"Loaded DataFrame with shape: {df.shape}")

# --- Start H2O and convert to H2OFrame ---
h2o.init()
hf = h2o.H2OFrame(df)
hf['Person'] = hf['Person'].asfactor()

# --- Split into train/test ---
train, test = hf.split_frame(ratios=[0.8], seed=1)

# Initialize XRT model (using histogram_type='Random' to enable XRT behavior)
xrt = H2ORandomForestEstimator(
    ntrees=100,
    max_depth=5,
    histogram_type="Random",
    seed=1
)

# Train the model on the entire dataset
xrt.train(y='Person', training_frame=hf)

# Print model performance on the full dataset
performance = xrt.model_performance(hf)
print(performance)