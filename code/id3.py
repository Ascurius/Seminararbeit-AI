import pandas as pd
import numpy as np
import json
from collections import Counter

def load_csv(file):
    data = pd.read_csv(file, header=0, sep=";")
    data = data.set_index("DAY")
    return data

def plot_tree(tree):
    plot = json.dumps(tree, indent=4)
    print(plot)
    return plot

def unique(list):

    list_unique = set(list)
    list_unique_sorted = sorted(list_unique)

    return list_unique_sorted

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

def calculate_all_IG(data_set, target_attribute):
    total_IG = {}
    for attribute in data_set.keys():
        if attribute != target_attribute:
            total_IG[attribute] = calculate_information_gain(data_set, attribute, target_attribute)
    return total_IG

def most_common_value(list):
    values, count = np.unique(list, return_counts=True)
    total = dict(zip(values, count))
    return max(total, key=lambda k: total[k])

def ID3(data_set, target_attribute, attributes):
    attributes = list(attributes)
    IG = calculate_all_IG(data_set, target_attribute)
    root = max(IG, key=lambda k: IG[k])

    if len(np.unique(data_set[target_attribute])) == 1:
        return np.unique(data_set[target_attribute])[0]
    if len(attributes) == 0:
        return root
    
    highest_IG_attribute = root
    tree = {highest_IG_attribute: {}}
    attributes.remove(highest_IG_attribute)

    for value in np.unique(data_set[highest_IG_attribute]):
        subset = data_set[data_set[highest_IG_attribute] == value]
        if subset.empty:
            tree[highest_IG_attribute][value] = most_common_value(data_set[target_attribute])
        else:
            subtree = ID3(subset, target_attribute, attributes)
            tree[highest_IG_attribute][value] = subtree
    return tree

S = load_csv("code/Weather.csv")

print(unique(S["TEMP"]))

#Tree = ID3(S, "PLAY", S.keys())

#plot_tree(Tree)