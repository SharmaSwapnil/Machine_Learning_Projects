# standard Imports
import pandas as pd
import numpy as np

# Standard sklearn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Reading the data
LS = pd.read_csv("LeslieSalt.csv")

# Creating the feaatures and labels
X = LS.drop("Price",axis=1)
y = LS["Price"]

# Instantiate the model
Rgr = RandomForestRegressor()
Rgr.fit(X,y)

X_tbp = pd.DataFrame({"County":[1],
					"Size":[246.8],
					"Elevation":[6],
					"Sewer":[1884],
					"Date":[0],
					"Flood":[1],
					"Distance":[2.5]})

y_preds_Price  = Rgr.predict(X_tbp)

print("The price estimated will be :",y_preds_Price)



