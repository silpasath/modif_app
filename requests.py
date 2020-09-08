import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'tenure':2, 'MonthlyCharges':1000, 'TotalCharges':1000})

print(r.json())