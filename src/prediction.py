#1 load scaler.pk1 and model.pk1 files
#create a function to predict
#1.load scaler.pkl and model.pkl
#2.create a function to predict
import pickle
import numpy as np
class insurance:
    def __init__(self):
        with open("artifacts/scaler.pkl","rb") as f:
            self.scaler=pickle.load(f)
        with open("artifacts/model.pkl","rb") as f:
            self.model=pickle.load(f)
    def prediction1(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
      input= np.array([[Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]])
      scaled_input=self.scaler.transform(input)
      result=self.model.predict(scaled_input)
      return result[0]