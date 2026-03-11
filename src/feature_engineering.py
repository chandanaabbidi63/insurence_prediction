#load the data
#create a model and train the model
#save the model in artifacts folders

import pandas as pd
from data_preprocessing import load_and_Split_data
from sklearn.preprocessing import StandardScaler
import pickle
x_train,x_test,y_train,y_test=load_and_Split_data()
print(x_train,y_train,x_test,y_test)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(x_train)
X_test_scaled=scaler.transform(x_test)
pd.DataFrame(X_train_scaled).to_csv("../data/processed/X_train_scaled.csv",index=False)
pd.DataFrame(X_test_scaled).to_csv("../data/processed/X_test_scaled.csv",index=False)
pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv",index=False)
pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv",index=False)
with open("../artifacts/scaler.pkl","wb") as f:
    pickle.dump(scaler,f)