import json
import requests
import pandas as pd
import mlflow
port = 5000
if __name__ == '__main__':
    df = pd.read_csv ('heart.csv')
    X = df.drop(['target'], axis=1)
#    print(X)
    D=X.iloc[:2]
#    print(D)
    input_data = D.to_json(orient="split")
#    print(input_data)
    endpoint = "http://127.0.0.1:{}/invocations".format(port)
    headers = {"Content-type": "application/json; format=pandas-records"}
#    print(headers)
    prediction = requests.post(endpoint, json=json.loads(input_data))
    print(prediction.text)
