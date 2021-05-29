import pandas as pd
import numpy as np
import json
from collections import Counter
from pprint import pprint

def load_csv(file):
    data = pd.read_csv(file, header=0, sep=";")
    if "Unnamed: 0" in data.keys():
        del data["Unnamed: 0"]
    return data

def plot_tree(tree):
    with open("code/tree.log", "w") as file:
        pprint(tree, file)
    return tree

def split_data(data_set, split_point):
    training_data_set = data_set.sample(frac=split_point)
    test_data_set = data_set.drop(training_data_set.index)
    return training_data_set, test_data_set

def entropy(attribute):
    entropy = 0.0
    values, count = np.unique(attribute, return_counts=True)
    for index in range(len(values)):
        probablility = count[index] / sum(count)
        entropy += (-probablility * np.log2(probablility))
    return entropy

def information_gain(data_set, attribute, target_attribute):
    values, count = np.unique(data_set[attribute], return_counts=True)
    entropy_val = 0.0
    
    for index, value in enumerate(values):
        subset = data_set[data_set[attribute] == value][target_attribute]
        probablility = count[index] / sum(count)
        entropy_val += ( probablility * entropy(subset) )
    target_entropy = entropy(data_set[target_attribute])
    information_gain = target_entropy - entropy_val

    return information_gain

def calculate_all_IG(data_set, target_attribute, attributes):
    total_IG = {}
    for attribute in attributes:
        if attribute != target_attribute:
            total_IG[attribute] = information_gain(data_set, attribute, target_attribute)
    return total_IG

def modal(list):
    values, count = np.unique(list, return_counts=True)
    total = dict(zip(values, count))
    return max(total, key=lambda k: total[k])

def ID3(data_set, target_attribute, attributes):
    if len(np.unique(data_set[target_attribute])) <= 1:
        return np.unique(data_set[target_attribute])[0]
    if len(attributes) <= 1:       
        return modal(data_set[target_attribute])

    IG = calculate_all_IG(data_set, target_attribute, attributes)
    best_attribute = max(IG, key=lambda k: IG[k])
    tree = {best_attribute: {}}
    attributes = [x for x in attributes if x != best_attribute]
    
    for value in np.unique(data_set[best_attribute]):
        subset = data_set[data_set[best_attribute] == value]
        if subset.empty:
            modal_value = modal(data_set[target_attribute])
            tree[best_attribute][value] = modal_value
        else:
            subtree = ID3(subset, target_attribute, attributes)
            tree[best_attribute][value] = subtree
    return tree

S = load_csv("code/data/Weather.csv")

#test_data_set = pd.concat([S, training_data_set, training_data_set]).drop_duplicates(keep=False)

#Tree = ID3(training_data_set, "RISK", training_data_set.columns)
S = S[S["WEATHER"] == "Rainy"]
print(calculate_all_IG(S, "PLAY", S.columns))

#plot_tree(tree)