# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex7 import *

# Set up filepaths
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")

# Imports
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # ✅ added

# Load data
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)

# Target
y = home_data.SalePrice

# Features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Train on split data → evaluate
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# Train on FULL data → final model
rf_model_on_full_data = RandomForestRegressor(random_state=1)  # ✅ fixed
rf_model_on_full_data.fit(X, y)

# Test data
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]              # ✅ fixed

# Predictions
test_preds = rf_model_on_full_data.predict(test_X)  # ✅ completed

# Save predictions
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)