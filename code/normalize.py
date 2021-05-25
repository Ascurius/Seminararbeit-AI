import random
import pandas as pd
import numpy as np

file = "code/RiskSample.csv"

df = pd.read_csv(file, sep=";")

def normalize_age(data):

    for idx, val in enumerate(data["AGE"]):
        if val <= 30:
            data["AGE"][idx] = "Young"
        elif (val > 30) and (val <= 40):
            data["AGE"][idx] = "Middle"
        elif val > 40:
            data["AGE"][idx] = "Old"

    for idx, val in enumerate(data["INCOME"]):
        if val <= 20000:
            data["INCOME"][idx] = "Low"
        elif (val > 20000) and (val <= 30000):
            data["INCOME"][idx] = "Middle"
        elif (val > 30000) and (val <= 45000):
            data["INCOME"][idx] = "High"
        elif (val > 45000) and (val < 60000):
            data["INCOME"][idx] = "Very High"
    
    del data["ID"]

    data.to_csv("code/RiskSampleNormalized.csv", sep=";")

normalize_age(df)
