from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from math import log2
from graphviz import Digraph
import io

app = Flask(__name__)
# ------------------------------------------CÂY QUYẾT ĐỊNH - GAIN --------------------------------------------------------------------#
# Hàm tính entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = -sum((counts[i] / np.sum(counts)) * log2(counts[i] / np.sum(counts)) for i in range(len(elements)))
    return entropy_value

# Hàm tính information gain
def info_gain(data, split_attribute, target_name):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = sum((counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute] == vals[i]).dropna()[target_name]) for i in range(len(vals)))
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Thuật toán ID3 tạo cây và xuất ra DOT format
def id3(data, original_data, features, target_attribute_name=None, parent_node_class=None, graph=None, parent_name=None):
    if target_attribute_name is None:
        target_attribute_name = data.columns[-1]

    if len(np.unique(data[target_attribute_name])) <= 1:
        graph.node(parent_name, label=np.unique(data[target_attribute_name])[0], shape='box')
        return np.unique(data[target_attribute_name])[0]

    if len(features) == 0:
        graph.node(parent_name, label=parent_node_class, shape='box')
        return parent_node_class

    parent_node_class = data[target_attribute_name].mode()[0]

    # Chọn thuộc tính có information gain cao nhất
    ig_values = [info_gain(data, feature, target_attribute_name) for feature in features]
    best_feature = features[np.argmax(ig_values)]

    graph.node(parent_name, label=best_feature, shape='ellipse')

    remaining_features = [feature for feature in features if feature != best_feature]

    # Tạo nhánh con cho mỗi giá trị của thuộc tính được chọn
    for value in np.unique(data[best_feature]):
        child_name = f"{parent_name}_{value}"
        graph.node(child_name, label=str(value), shape='ellipse')
        graph.edge(parent_name, child_name)
        sub_data = data[data[best_feature] == value]
        id3(sub_data, original_data, remaining_features, target_attribute_name, parent_node_class, graph, child_name)

    return parent_node_class

# Tạo hàm vẽ cây
def visualize_tree(data, features, target_column="Play"):
    graph = Digraph(strict=True)
    graph.attr(dpi='300')  # Optional: sets the resolution of the image

    tree = id3(data, data, features, target_column, graph=graph, parent_name='root')

    return graph

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        action = request.form['action']
        if file:
            if action == 'decision_tree':
                # Đọc file CSV
                df = pd.read_csv(file)
                # Giả sử cột cuối cùng là target
                features = df.columns[:-1].tolist()  # Các cột feature
                target_column = df.columns[-1]  # Cột target

                # Vẽ cây quyết định
                graph = visualize_tree(df, features, target_column)
                
                # Lưu cây vào tệp .png
                graph.render('decision_tree', format='png')

                # Trả về hình ảnh cây quyết định
                return send_file('decision_tree.png', mimetype='image/png')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
