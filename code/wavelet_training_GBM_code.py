import pandas as pd
import numpy as np
import pandas as pd
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

# --- Read Excel file to DataFrame ---
input_excel_path = "./wavelet_features_reduced.xlsx"
df = pd.read_excel(input_excel_path)

h2o.init()
hf = h2o.H2OFrame(df)
hf['Person'] = hf['Person'].asfactor()

# --- Split into train/test ---
train, test = hf.split_frame(ratios=[0.8], seed=1)

from h2o.estimators.gbm import H2OGradientBoostingEstimator

# Initialize GBM model
gbm = H2OGradientBoostingEstimator(
    ntrees=100,
    max_depth=5,
    learn_rate=0.1,
    seed=1
)

# Train the model
gbm.train(y='Person', training_frame=train)

# Print model performance on training data
performance = gbm.model_performance(train=True)
print(performance)