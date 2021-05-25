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
    training_data_set = S.sample(frac=split_point)
    test_data_set = S.drop(training_data_set.index)
    return training_data_set, test_data_set

def unique(iterable):

    list_unique = set(iterable)
    list_unique_sorted = sorted(list_unique)
    list_unique_sorted_count = {}

    for unique_value in list_unique_sorted:
        count = 0
        for value in iterable:
            if value == unique_value:
                count += 1
        list_unique_sorted_count[unique_value] = count

    return list_unique_sorted_count

def calculate_entropy(attribute):
    entropy = 0
    _, count = np.unique(attribute, return_counts=True)
    for index in range(len(count)):
        probablility = count[index] / sum(count)
        entropy += (-probablility * np.log2(probablility))
    return entropy

def calculate_information_gain(data_set, attribute, target_attribute):
    values, count = np.unique(data_set[attribute], return_counts=True)
    entropy = 0
    
    for index, value in enumerate(values):
        subset = data_set[data_set[attribute] == value][target_attribute]
        probablility = count[index] / sum(count)
        entropy += ( probablility * calculate_entropy(subset) )
    information_gain = calculate_entropy(data_set[target_attribute]) - entropy

    return information_gain

def calculate_all_IG(data_set, target_attribute, attributes):
    total_IG = {}
    for attribute in attributes:
        if attribute != target_attribute:
            total_IG[attribute] = calculate_information_gain(data_set, attribute, target_attribute)
    return total_IG

def most_common_value(list):
    values, count = np.unique(list, return_counts=True)
    total = dict(zip(values, count))
    return max(total, key=lambda k: total[k])

def ID3(data_set, target_attribute, attributes):
    if len(np.unique(data_set[target_attribute])) <= 1:
        return np.unique(data_set[target_attribute])[0]
    if len(attributes) <= 1:       
        return most_common_value(data_set[target_attribute])

    IG = calculate_all_IG(data_set, target_attribute, attributes)
    root = max(IG, key=lambda k: IG[k])
    highest_IG_attribute = root
    tree = {highest_IG_attribute: {}}
    attributes = [x for x in attributes if x != highest_IG_attribute]

    for value in np.unique(data_set[highest_IG_attribute]):
        subset = data_set[data_set[highest_IG_attribute] == value]
        if subset.empty:
            tree[highest_IG_attribute][value] = most_common_value(data_set[target_attribute])
        else:
            subtree = ID3(subset, target_attribute, attributes)
            tree[highest_IG_attribute][value] = subtree
    return tree

S = load_csv("code/RiskSampleNormalized-Ausschnitt.csv")
#S = load_csv("code/Weather.csv")

training_data_set = split_data(S, 0.3)

Tree = ID3(training_data_set, "RISK", training_data_set.columns)

plot_tree(Tree)