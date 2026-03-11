#load processed data ffrom processed folder
#create a model and train the model
#save the model in artifacts folders
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
X_train=pd.read_csv("../data/processed/X_train_scaled.csv")
X_test=pd.read_csv("../data/processed/X_test_scaled.csv")
y_train=pd.read_csv("../data/processed/y_train.csv")
y_test=pd.read_csv("../data/processed/y_test.csv")
print(X_train)
model=LinearRegression()
model.fit(X_train,y_train)
with open("../artifacts/model.pkl","wb") as f:
    pickle.dump(model,f)
    