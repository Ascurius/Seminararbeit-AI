import random
import pandas as pd
import numpy as np

file = "code/RiskSample.csv"
file1 = "code/RiskSamplePreview.csv"

df = pd.read_csv(file1, sep=";")

def normalize(data):

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

    for idx, val in enumerate(data["NUMKIDS"]):
        if val == 0.0:
            data["NUMKIDS"][idx] = "No"
        else:
            data["NUMKIDS"][idx] = "Yes"
    
    for idx, val in enumerate(data["LOANS"]):
        if val == 0.0:
            data["LOANS"][idx] = "No"
        else:
            data["LOANS"][idx] = "Yes"

    data.rename(columns={"NUMKIDS": "HASKIDS"})
    #del data["ID"]

    data.to_csv("code/RiskSampleNormalizedPreview.csv", sep=";")

normalize(df)