import pandas as pd
import numpy as np

def load_csv(file):
    data = pd.read_csv(file, header=0, sep=";")
    data = data.set_index("DAY")
    return data

def calculate_entropy(attribute):
    entropy = 0
    _, count = np.unique(attribute, return_counts=True)
    for index in range(len(count)):
        probablility = count[index] / sum(count)
        entropy += (-probablility * np.log2(probablility))
    return entropy

def calculate_information_gain(data_set, attribute, classifier):
    values, count = np.unique(data_set[attribute], return_counts=True)
    entropy = 0
    
    for index, value in enumerate(values):
        subset = data_set[data_set[attribute] == value][classifier]
        probablility = count[index] / sum(count)
        entropy += ( probablility * calculate_entropy(subset) )
    information_gain = calculate_entropy(data_set[classifier]) - entropy

    return information_gain

S = load_csv("code/Weather.csv")


print("IG: " + str(calculate_information_gain(S, "HUMIDITY", "PLAY")))