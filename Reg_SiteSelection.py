## Predicting Sales for a store location

# 1.Standard Imports
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 2. Load the data
sunflower_customers = pd.read_csv("data/SiteSelection.csv")




# 3. Split the data into Features X and labels y
X=sunflower_customers.drop("Annual Sales",axis=1)
y=sunflower_customers["Annual Sales"] 
print(y)


# 4. Split the data into train and test
# from sklearn.model_selection import train_test_split
# np.random.seed(0)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# 5. Build the model
from sklearn.ensemble import RandomForestRegressor
reg_model = RandomForestRegressor()
reg_model.fit(X,y)

# 6. Make Predictions
X_to_be_predicted  = pd.DataFrame({"Store":[14],
									"Profiled Customers":[4]
									})


y_preds =  reg_model.predict(X_to_be_predicted)
print("Following are the predictions of sales : ",y_preds)

# Calculate the accuracy score on test
accuracy_score = reg_model.r2_score(X_test,y_test)

print("Accuracy score of the model is :",accuracy_score)



