from joblib import load 
model=load("Crop.joblib")
feature=([[0.25421252, 0.27579609, 0.01636317, 3.55194693, 0.95636671,
       0.64209145, 0.14771821]])
result=model.predict(feature)
print("Result: ",result[0])
