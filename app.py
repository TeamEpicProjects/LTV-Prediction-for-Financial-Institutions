import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv
import codecs
#from LTV import LTV_prediction
import pickle

app = FastAPI()



@app.get('/')
def index():
    return {'message': 'Welcome to LTV Prediction System'}

# @app.get("/data")
# def data():
#     df = pd.read_csv('C:\Users\Sarth\Downloads\final_combined.csv')
#     df.dropna(inplace=True)
#     return {'dataframe': f'{df}'}



@app.post('/predict')
async def predict_LTV(ID: int, amount_financed: float, asset_cost: float, tenure: float, payment_mode: str, installment_mode: str):

    # the dataset and the model
    df = pd.read_csv('data_final.csv')


    df.drop('ID', axis=1, inplace=True)
    # df1.drop(['ID', '#TypeOfLoans', 'LTV', 'Top-up Month', 'Bank-lists'], axis=1, inplace=True)
    pickle_in = open('lgbm_model.pkl', 'rb')
    model = pickle.load(pickle_in)
   

    # LabelEncoding
    if payment_mode == 'Direct Debit':
        payment_mode = 3
    elif payment_mode == 'ECS':
        payment_mode = 4
    elif payment_mode == 'PDC':
        payment_mode = 7
    elif payment_mode == 'Billed':
        payment_mode = 1
    elif payment_mode == 'PDC_E':
        payment_mode = 9
    elif payment_mode == 'Auto Debit':
        payment_mode = 0
    elif payment_mode == 'SI Reject':
        payment_mode = 10
    elif payment_mode == 'Cheque':
        payment_mode = 2
    elif payment_mode == 'ECS Reject':
        payment_mode = 5
    elif payment_mode == 'Escrow':
        payment_mode = 6
    elif payment_mode == 'PDC Reject':
        payment_mode = 8
   
    if installment_mode == 'Advance':
        installment_mode = 0
    elif installment_mode == 'Arrear':
        installment_mode = 1



    pred = df.loc[ID:ID,:]

    pred.loc[:,'PaymentMode'] = payment_mode
    pred.loc[:,'InstlmentMode'] = installment_mode
    pred.loc[:,'AssetCost'] = asset_cost
    pred.loc[:,'Tenure'] = tenure
    # pred.drop(['ID', '#TypeOfLoans', 'LTV', 'Top-up Month', 'Bank-lists'], axis=1, inplace=True)

    # prediction 
    prediction = model.predict(pred)
    # prediction = print(pred)
    return {'prediction': f'{prediction}'}





if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)



