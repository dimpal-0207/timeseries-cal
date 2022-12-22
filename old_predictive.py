import datetime
import decimal
import json
# https://www.kaggle.com/code/satishgunjal/tutorial-time-series-analysis-and-forecasting
from bson import json_util
from dateutil.relativedelta import relativedelta
from flask import Flask, jsonify, Response, request
from pymongo import MongoClient
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pymongo import MongoClient

client = MongoClient("mongodb+srv://kamalsherma:l2GIQc5mMOu0gtDo@cluster0.bpyxs.mongodb.net/greenalytics-testing-db")
# mongodb+srv://kamalsherma:l2giqc5mmou0gtdo@cluster0.bpyxs.mongodb.net/greenalytics-testing-db


print("===", client.list_database_names())
db = client['greenalytics-testing-db']

env_data = db.reportdatas
data = env_data.find()
data1 = list(data)

def Autoreg(arr):
    model = AutoReg(arr, lags=1, trend='t')
#     print("___", model)
    model_fit = model.fit()
    print("+++++---", model_fit)
    x = round(float(model_fit.predict(len(arr), len(arr))), 2)
    print("==========", x)
    return round(float(model_fit.predict(len(arr), len(arr))), 2)


# Moving-Average model
def MA(arr):
    model = ARIMA(list(arr), order=(0, 0, 1), trend='t')
    model_fit = model.fit()
    return round(float(model_fit.predict(len(arr), len(arr))), 2)


# Auto Regressive Moving-Average model
def AutoMA(arr):
    model = ARIMA(list(arr), order=(2, 0, 2), trend='t')
    model.initialize_approximate_diffuse() # this line
    model_fit = model.fit()
    return round(float(model_fit.predict(len(arr), len(arr))), 2)


# Seasonal Autoregressive Integrated Moving-Average model
def SAutoMA(arr):
    model = SARIMAX(list(arr), order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), trend='t')
    model_fit = model.fit(disp=False)
    return round(float(model_fit.predict(len(arr), len(arr))), 2)


YEAR_FORMATS = []


def flatten_nested_json_df(df):
    df = df.reset_index()
    s = (df.applymap(type) == list).all()
    list_columns = s[s].index.tolist()

    s = (df.applymap(type) == dict).all()
    dict_columns = s[s].index.tolist()

    while len(list_columns) > 0 or len(dict_columns) > 0:
        new_columns = []

        for col in dict_columns:
            horiz_exploded = pd.json_normalize(df[col]).add_prefix(f'{col}.')
            horiz_exploded.index = df.index
            df = pd.concat([df, horiz_exploded], axis=1).drop(columns=[col])
            new_columns.extend(horiz_exploded.columns)  # inplace

        for col in list_columns:
            # print(f"exploding: {col}")
            df = df.drop(columns=[col]).join(df[col].explode().to_frame())
            new_columns.append(col)

        s = (df[new_columns].applymap(type) == list).all()
        list_columns = s[s].index.tolist()

        s = (df[new_columns].applymap(type) == dict).all()
        dict_columns = s[s].index.tolist()
    return df


app = Flask(__name__, template_folder = 'templates')

def report_data():
    env_data = db.reportdatas
    data = env_data.find()
    data1 = list(data)
    results = pd.json_normalize(data1)
    df = pd.DataFrame(results)
    outdf = flatten_nested_json_df(df)
    # outdf

    df = df[['companyId', 'categoryId', 'locationId', 'subCategory1', 'subCategory2', 'subCategory3', 'subCategory4',
             'subCategory5', 'subCategory6', 'subCategory7', 'subCategory8', 'subCategory9', 'subCategory10',
             'category', 'typeOfData', 'dataPoints.ReadingValue.value', 'dataPoints.ReadingValue.units', 'year',
             'month', 'isConsolidatedData', 'formatedDate', 'createdAt', 'modifiedAt']]
    cols = ["year", "month"]
    df['date'] = df[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    df = df.drop(['month'], axis=1)
    # print(df.head(5))
    df['date'] = df[['date']]
    df.head(5)

    df['date'] = pd.to_datetime(df['date'], format='%Y-%b').dt.strftime("%m-%Y")
    df = df.dropna()
    df = df.rename(columns={'dataPoints.ReadingValue.value': 'valuedata', 'dataPoints.ReadingValue.units': 'units'})
    df = df.dropna(axis=0)
    df = df.drop(['year'], axis=1)
    # df.head(5)
    df = df.pivot_table(
        index=['companyId', 'categoryId', 'locationId', 'category', 'units', 'subCategory1', 'subCategory2',
               'subCategory3', 'subCategory4', 'subCategory5', 'subCategory6', 'subCategory7', 'subCategory8',
               'subCategory9', 'subCategory10', 'typeOfData', 'isConsolidatedData'], columns='date', values="valuedata",
        aggfunc='first')
    df = df.reset_index([0])
    df = df.fillna('0')

    return df


def predict(df, start=5, model="Autoreg", year=2):
    year = int(year)
    if year == 0:
        year = 2
    df_new = df.copy()
    year = year * 12
    print(year)
    for j in range(year):
        new_arr = []
        for i in range(len(df_new)):
            print("----->i", i)
            arr = []
            arr = np.array(df_new.iloc[i, start:])
            print("&&&&&&&&", arr)
            if model == "Autoreg":
                value = Autoreg(arr)
                print("#$#$#$#$", value)
            elif model == "MA":
                value = MA(arr)
            elif model == "AutoMA":
                value = AutoMA(arr)
            elif model == "SAutoMA":
                value = SAutoMA(arr)
            new_arr.append(value)
        # print("?>?>?>?>?>>", df_new.columns.tolist())
        try:
            dt = (datetime.datetime.strptime(df_new.columns[-1].replace("/", "-"), "%m-%Y") + relativedelta(
                months=1)).strftime('%m/%Y')
            # print("??????.......", dt)
        except Exception as e:
            try:
                dt = (datetime.datetime.strptime(df_new.columns[-1].replace("/", "-"), "%m-%y") + relativedelta(
                    months=1)).strftime('%m/%y')
            except Exception as e:
                dt = (datetime.datetime.strptime(df_new.columns[-1].replace("/", "-"), "%m-%y") + relativedelta(
                    months=1)).strftime('%m/%Y')
                # print("+%%%%____", dt)

        df_new[dt] = new_arr
        # print(">>>>>>>>....", df_new[dt])
    return df_new


@app.route('/reportdata_pred_autoreg/<year>', methods=['GET'])
def reportdata_pred_autoreg(year):
    companyId = request.form.get("companyId", None)
    print("====u_email", companyId)
    mongo_string = request.form.get("mongo_string", None)
    print("---mongo_string")
    years = []
    df = report_data()
    predict_mines_data={}
    for i in list(df.columns[5:]):
        years.append(i[-4:])
    years = list(set(years))
    print("@@@@", years)
    years.sort()
    years
    curr_year = years[-1]
    print("!!!!!!!", curr_year)
    df_new = predict(df, start=5, model="Autoreg", year=year)
    # print("*-*-*-", df_new)
    data = df_new.reset_index(inplace=True)
    data1 = df_new.reset_index()
    # print("===>df_new" ,df_new)
    df_new = df_new.melt(id_vars=['companyId', 'categoryId','locationId','category', 'subCategory1', 'subCategory2', 'subCategory3','subCategory4','subCategory5','subCategory6','subCategory7','subCategory8','subCategory9','subCategory10','typeOfData','units', 'isConsolidatedData'], var_name="date", value_name="Value")
    print("====>df_new", df_new.head(5))
    print("+#+#+#+#", df_new.tail(5))
    df_new['date'] = df_new['date'].str.replace("-", "/")
    df_new['year'] = pd.to_datetime(df_new['date']).dt.year
    df_new['year']= df_new['year'].astype(str)
    df_new['month'] = pd.to_datetime(df_new['date'], format='%m/%Y').dt.strftime('%b')

    # df_new['month_value'] = df_new['month'].astype(str) + ':' + df_new['Value'].astype(str)
    # df_new.head(5)
    # print(df_new.dtypes)
    # df_new.to_csv('my_pred_data')
    df_new = df_new[~(df_new['year'] < curr_year)]
    list1 = []
    cat = {}
    month = {}
    for data, value in df_new.iterrows():
        data_object = -1
        for index, ele in enumerate(list1):
            if ele['categoryId'] == value['categoryId'] and ele['locationId'] == value['locationId'] and ele['companyId'] == value['companyId'] and ele['category'] == value['category'] and ele['subCategory1'] == value['subCategory1'] and ele['subCategory2'] == value['subCategory2'] and ele['subCategory3'] == value['subCategory3'] and ele['subCategory4'] == value['subCategory4'] and ele['subCategory5'] == value['subCategory5'] and ele['subCategory6'] == value['subCategory6'] and ele['subCategory7'] == value['subCategory7']and ele['subCategory8'] == value['subCategory8'] and ele['subCategory9'] == value['subCategory9'] and ele['subCategory10'] == value['subCategory10'] and ele['typeOfData'] == value['typeOfData'] and ele['isConsolidatedData'] == value['isConsolidatedData'] and ele['units'] == value['units'] and ele['year'] == value['year']:
                data_object = index
                # print("+++++>count", index)
                # print("=====",ele)
                # print("data_object", data_object)
        if data_object == -1:
            cat = {**value}

            month = {}
            month[value.month] = value.Value
            # print("___>", month)
            cat['months'] = month
            # print("____cat", cat)
            list1.append(cat)

        else:
            list1[data_object]['months'][value.month] = value.Value
            # print("=====>l1", list1[data_object])
            # print("__a", data_object)  # list of all elements with .n==30
    # result = db.brsr_prediction.insert_many(list1)
    # print("___result", result)
    for val in list1:
        update_database = db.brsr_predictions.update_one({'categoryId': val['categoryId'],
                                                 'locationId': val['locationId'],
                                                 'companyId': val['companyId'],
                                                'category': val['category'],
                                                 'subCategory1': val['subCategory1'],
                                                 'subCategory2': val['subCategory2'],
                                                 'subCategory3': val['subCategory3'],
                                                  'subCategory4': val['subCategory4'],
                                                  'subCategory5': val['subCategory5'],
                                                  'subCategory6': val['subCategory6'],
                                                  'subCategory7': val['subCategory7'],
                                                  'subCategory8': val['subCategory8'],
                                                  'subCategory9': val['subCategory9'],
                                                  'subCategory10': val['subCategory10'],
                                                 'typeOfData': val['typeOfData'],
                                                'units': val['units'],
                                                'isConsolidatedData': val['isConsolidatedData'],
                                                'year' : val['year']},

                                                {'$set': {'months': val['months'], "modifiedAt": datetime.datetime.now()},
                                                 "$setOnInsert": {"createdAt": datetime.datetime.now()}
                                                 },
                                                upsert=True)

        print("====update database and create", update_database)

    return Response("updated all prediction done")


def carbon_data():
    env_data = db.carbondatas
    data = env_data.find()
    data1 = list(data)
    results = pd.json_normalize(data1)
    df1 = pd.DataFrame(results)
    outdf = flatten_nested_json_df(df1)
    df1 = df1[['companyId', 'category', 'factorName', 'factorName1', 'locationId', 'scope', 'subCategory1', 'subCategory2','subCategory3', 'subCategory4', 'subCategory5', 'subCategory6', 'subCategory7', 'subCategory8', 'subCategory9','emission_factor', 'fuel_quantity', 'year', 'month', 'units', 'value']]
    cols = ["year", "month"]
    df1['date'] = df1[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    df1 = df1.drop(['month'], axis=1)
    df1['date'] = df1[['date']]
    df1.head(5)
    df1['date'] = pd.to_datetime(df1['date'], format='%Y-%b').dt.strftime("%m-%Y")
    df1 = df1.dropna()
    df1 = df1.pivot_table(
        index=['companyId', 'locationId', 'category', 'factorName', 'factorName1', 'scope', 'subCategory1',
               'subCategory2', 'subCategory3', 'subCategory4', 'subCategory5', 'subCategory6', 'subCategory7',
               'subCategory8', 'subCategory9', 'emission_factor', 'fuel_quantity', 'units'], columns='date',
        values="value")
    df1 = df1.reset_index([0])
    print("=====>head", df1.head(5))
    print("=====>tail", df1.tail(5))


    # # df = df.isna().sum()
    df1 = df1.fillna('0')
    return df1


@app.route('/carbon_data_pred_autoreg/<year>', methods=['GET'])
def carbon_pred_autoreg(year):
    companyId = request.form.get("companyId", None)
    print("====u_email", companyId)
    mongo_string = request.form.get("mongo_string", None)
    years = []
    df = carbon_data()
    for i in list(df.columns[5:]):
        years.append(i[-4:])
    years = list(set(years))
    print("@@@@", years)
    years.sort()
    years
    curr_year = years[-1]
    print("!!!!!!!", curr_year)
    df_new = predict(df, start=2, model="Autoreg", year=year)
    data = df_new.reset_index(inplace=True)
    data1 = df_new.reset_index()
    df_new = df_new.melt(
        id_vars=['locationId', 'factorName', 'factorName1', 'companyId', 'scope', 'category', 'subCategory1',
                 'subCategory2', 'subCategory3', 'subCategory4', 'subCategory5', 'subCategory6', 'subCategory7',
                 'subCategory8', 'subCategory9', 'units', 'emission_factor', 'fuel_quantity'], var_name="date",
        value_name="Value")

    df_new['date'] = df_new['date'].str.replace("-", "/")
    df_new['year'] = pd.to_datetime(df_new['date']).dt.year
    print(df_new.head(5))
    df_new['year'] = pd.to_datetime(df_new['date']).dt.year
    df_new['year'] = df_new['year'].astype(str)
    df_new['month'] = pd.to_datetime(df_new['date'], format='%m/%Y').dt.strftime('%b')
    df_new['month_value'] = df_new['month'].astype(str) + ':' + df_new['Value'].astype(str)
    df_new.head(5)
    print(df_new.dtypes)
    df_new.to_csv('my_pred_data')
    df_new = df_new[~(df_new['year'] < curr_year)]
    list1 = []
    cat = {}
    month = {}
    for data, value in df_new.iterrows():
        data_object = -1
        for index, ele in enumerate(list1):
            if ele['category'] == value['category'] and ele['locationId'] == value['locationId'] and ele[
                'companyId'] == value['companyId'] and ele['factorName'] == value['factorName'] and ele['factorName1'] == value['factorName1'] and ele['subCategory1'] == \
                    value['subCategory1'] and ele['subCategory2'] == value['subCategory2'] and ele['subCategory3'] == \
                    value['subCategory3'] and ele['subCategory4'] == value['subCategory4'] and ele['subCategory5'] == \
                    value['subCategory5'] and ele['subCategory6'] == value['subCategory6'] and ele['subCategory7'] == \
                    value['subCategory7'] and ele['subCategory8'] == value['subCategory8'] and ele['subCategory9'] == \
                    value['subCategory9'] and ele['emission_factor'] == value['emission_factor'] and ele['fuel_quantity'] == \
                    value['fuel_quantity'] and ele['scope'] == value['scope'] and ele['units'] == \
                    value['units'] and ele['year'] == value['year']:
                data_object = index
                # print("+++++>count", index)
                # print("=====",ele)
                # print("data_object", data_object)
        if data_object == -1:
            cat = {**value}

            month = {}
            month[value.month] = value.Value
            # print("___>", month)
            cat['months'] = month
            # print("____cat", cat)
            list1.append(cat)

        else:
            list1[data_object]['months'][value.month] = value.Value
            # print("=====>l1", list1[data_object])
            # print("__a", data_object)  # list of all elements with .n==30
    # result = db.brsr_prediction.insert_many(list1)
    # print("___result", result)
    for val in list1:
        update_database = db.carbon_predictions.update_one({'companyId': val['companyId'],
                                                          'locationId': val['locationId'],
                                                          'factorName': val['factorName'],
                                                            'factorName1': val['factorName1'],
                                                          'category': val['category'],
                                                          'subCategory1': val['subCategory1'],
                                                          'subCategory2': val['subCategory2'],
                                                          'subCategory3': val['subCategory3'],
                                                          'subCategory4': val['subCategory4'],
                                                          'subCategory5': val['subCategory5'],
                                                          'subCategory6': val['subCategory6'],
                                                          'subCategory7': val['subCategory7'],
                                                          'subCategory8': val['subCategory8'],
                                                          'subCategory9': val['subCategory9'],
                                                          'emission_factor': val['emission_factor'],
                                                            'fuel_quantity': val['fuel_quantity'],
                                                          'scope': val['scope'],
                                                          'units': val['units'],

                                                          'year': val['year']},

                                                         {'$set': {'months': val['months'],
                                                                   "modifiedAt": datetime.datetime.now()},
                                                          "$setOnInsert": {"createdAt": datetime.datetime.now()}
                                                          },
                                                         upsert=True)

        print("====update database and create", update_database)

    return Response("updated all prediction done")



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
