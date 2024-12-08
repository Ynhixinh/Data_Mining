from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from flask import jsonify

eps = np.finfo(float).eps
from numpy import log2 as log

app = Flask(__name__)
# Đảm bảo thư mục lưu trữ media tồn tại
if not os.path.exists('media'):
    os.makedirs('media')


df = pd.read_csv('data.csv')

# 1. Tính entropy của dataset
entropy_node = 0
values = df['Play'].unique()
for value in values:
    fraction = df['Play'].value_counts(normalize= True) #đếm số lần xuất hiện của Yes và No -> value_count, normalize ->tính % xuất hiện
    entropy_node += -(fraction*np.log2(fraction))

# print(f'Values: {values}')
# print (f'entropy_node : {entropy_node}')

#2. Viết hàm tính entropy từng thuộc tính
def  ent (df, attribute):
    target_variables = df['Play'].unique()
    variables = df[attribute].unique()

    entropy_attribute = 0
    for variable in variables:
        entropy_each_feature = 0
        for target_variable  in target_variables:
            num = len(df[attribute][df[attribute]== variable][df['Play']== target_variable]) #vd cột weather có Sunny,Rain -> đếm số dòng dữ liệu của Weather có Sunny mà Yes và No
            den = len(df[attribute][df[attribute]== variable]) #đếm số dòng của thuộc tính (tổng dòng dữ liệu của data)
            fraction = num/(den+eps)
            entropy_each_feature += -fraction*log(fraction+eps) #Tính entropy của từng specific values trong cột Weather (ví dụ)
        fraction2 = den / len(df) #den = tổng số dòng weather là sunny  / len = số dòng dữ liệu của dataset
        entropy_attribute += -fraction2*entropy_each_feature #Tổng entropy của cột Weather = cộng dồn sau mỗi giá trị cụ thể của cột Weather

    return (abs(round(entropy_attribute,4)))

# for column in df.columns:
#     if column != 'Play':
#         print(f'{column}: {ent(df,column)}')

#1. Tìm entropy cho cột cuối cùng (cột mục tiêu)
def find_entropy(df):
    Class = df.keys()[-1] #danh sách các tên cột
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        entropy += -(fraction * np.log2(fraction))
    return entropy

#2.Tính entropy cho từng cột
def find_entropy_attribute (df, attribute):
    Class = df.keys()[-1]
    target_variables = df[Class].unique()
    variables = df[attribute].unique()

    entropy2= 0
    for variable in variables:
        entropy = 0
        for target_variable  in target_variables:
            num = len(df[attribute][df[attribute]== variable][df[Class]== target_variable]) #vd cột weather có Sunny,Rain -> đếm số dòng dữ liệu của Weather có Sunny mà Yes và No
            den = len(df[attribute][df[attribute]== variable]) #đếm số dòng của thuộc tính (tổng dòng dữ liệu của data)
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps) #Tính entropy của từng specific values trong cột Weather (ví dụ)
        fraction2 = den / len(df) #den = tổng số dòng weather là sunny  / len = số dòng dữ liệu của dataset
        entropy2 += -fraction2*entropy #Tổng entropy của cột Weather = cộng dồn sau mỗi giá trị cụ thể của cột Weather

    return (abs(round(entropy2,4)))


#3. Tìm thuộc tính info thấp nhất
def find_winner (df):
    Entropy_att = []
    Inf = []

    for key in df.keys()[:-1]:
        Inf.append(find_entropy(df)-find_entropy_attribute(df,key)) #Tính information từng attribute rồi thêm vào mảng Inf
    
    return df.keys()[:-1][np.argmax(Inf)] #Tìm thuộc tính có info thấp nhất (gain mới lấy cao nhất)

def get_subtable (df,node, value):
    return df[df[node]== value].reset_index(drop=True) #lọc ra các giá trị của cột được chọn từ hàm find_winner

#4. Xây dựng ây quyết định
def build_tree (df, tree = None):
    Class = df.keys()[-1]

    #Tìm node
    node = find_winner(df)

    #Lấy ra những giá trị của cột node
    att_value = np.unique(df[node])

    #Tạo dic rỗng để tạo cây
    if tree is None:
        tree = {}
        tree[node] = {}

    #Tạo vòng lặp -> gọi đệ quy
    for value in att_value:

        subtable = get_subtable(df,node,value)
        clValue, counts = np.unique(subtable[Class], return_counts = True)

        if len(counts) == 1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = build_tree(subtable)

    return tree

# t = build_tree(df)
# import pprint
# pprint.pprint(t)


#5. Vẽ cây
import pydot
import uuid
def generate_unique_node():
    """ Generate a unique node label"""
    return str(uuid.uuid1())

def create_node (graph, label, shape = 'oval'):
    node = pydot.Node(generate_unique_node(), label = label, shape = shape)
    graph.add_node(node)
    return node

def create_edge(graph, node_parent, node_child, label):
    link = pydot.Edge(node_parent,node_child, label = label)
    graph.add_edge(link)
    return link

def walk_tree (graph, dictionary, prev_node = None):
    """Recursive construction of a decision tree stored as a dictionary"""

    for parent, child in dictionary.items():
        #root
        if not prev_node:
            root = create_node(graph,parent)
            walk_tree(graph,child,root)
            continue

        #node
        if isinstance(child, dict):
            for p,c  in child.items():
                n = create_node(graph,p)
                create_edge(graph, prev_node, n, str(parent))
                walk_tree(graph,c,n)

        #leaf
        else:
            leaf = create_node(graph, str(child), shape='box')
            create_edge(graph, prev_node, leaf, str(parent))


def plot_tree (dictionary, filename = "media/DecisionTree3.png"):
    graph = pydot.Dot(graph_type = 'graph')
    walk_tree(graph, tree)
    graph.write_png(filename)

tree = build_tree(df, tree= None)

graph = pydot.Dot(graph_type = 'digraph')
walk_tree(graph, tree)


#-------------------------------------------K-Means Gom cụm ------------------------------------------------------------------#
# Khởi tạo lớp KMeans
class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def initialize_specific_clusters(self, X, initial_clusters):
        self.labels = np.zeros(len(X), dtype=int)

        # Gán nhãn theo phân hoạch ban đầu
        for cluster_id, points in initial_clusters.items():
            for point_idx in points:
                self.labels[point_idx] = cluster_id

        # Tính centroids ban đầu
        self.centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                self.centroids[i] = np.mean(cluster_points, axis=0)

        return self.labels

    def fit(self, X):
        for iteration in range(self.max_iters):
            old_centroids = self.centroids.copy()
            old_labels = self.labels.copy()

            # Tính khoảng cách từ mỗi điểm đến các centroids
            distances = np.zeros((X.shape[0], self.n_clusters))  # Tạo mảng lưu khoảng cách
            for i in range(self.n_clusters):
                # Tính khoảng cách Euclidean giữa điểm và centroid i
                distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)

            # Gán nhãn mới
            self.labels = np.argmin(distances, axis=1)

            # Cập nhật centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels == k) > 0:
                    self.centroids[k] = X[self.labels == k].mean(axis=0)

            # Kiểm tra hội tụ
            if np.all(old_centroids == self.centroids) and np.all(old_labels == self.labels):
                print(f"Hội tụ sau {iteration + 1} vòng lặp")
                break

            # In thông tin mỗi vòng lặp
            print(f"\nVòng lặp {iteration + 1}:")
            print("Centroids:", self.centroids)
            for k in range(self.n_clusters):
                cluster_points = X[self.labels == k]
                print(f"Cụm {k + 1}: {[f'A{i+1}' for i, label in enumerate(self.labels) if label == k]}")

        return self.labels

#----------------------------------------------------------------------------------------------------------------------------#
MEDIA_FOLDER = os.path.join(os.getcwd(), 'media')

@app.route('/media/<filename>')
def get_image(filename):
    return send_from_directory(MEDIA_FOLDER, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            if 'decision_tree' in request.form:
                tree = build_tree(df)
                plot_tree(tree)
                return render_template("index.html", image="DecisionTree3.png")
            
            elif 'K-means' in request.form:
                # chuyển thành số
                le = LabelEncoder()
                df_encoded = df.apply(le.fit_transform)

                # Lấy 3 thuộc tính đầu tiên
                data = df_encoded.iloc[:, :3].to_numpy()

                # Khởi tạo điểm (dòng dữ liệu) thuộc các cụm
                initial_clusters = {
                    0: [0, 1, 2, 3, 4],
                    1: [5, 6, 7, 8],       
                    2: [9, 10, 11, 12]     
                }

                # Khởi tạo KMeans
                kmeans = KMeans(n_clusters=3)
                initial_labels = kmeans.initialize_specific_clusters(data, initial_clusters)
                final_labels = kmeans.fit(data)

                # # Lưu hình ảnh kết quả phân cụm
                # plt.figure(figsize=(8, 6))
                # colors = ['r', 'g', 'b']
                # for i in range(len(data)):
                #     plt.scatter(data[i, 0], data[i, 1], c=colors[kmeans.labels_[i]], s=100)
                # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='*', s=200)
                # plt.title("KMeans Clustering")
                # plt.xlabel("Feature 1")
                # plt.ylabel("Feature 2")
                # plt.grid(True)

                # # Lưu kết quả đồ thị vào file trong thư mục media
                # image_path = os.path.join(MEDIA_FOLDER, "KMeansResult.png")
                # plt.savefig(image_path)
                # plt.close()

                # Truyền kết quả phân cụm vào template
                clusters_info = {}
                for k in range(kmeans.n_clusters):
                    cluster_points = [f'A{i+1}' for i, label in enumerate(kmeans.labels_) if label == k]
                    clusters_info[f"Cụm {k+1}"] = cluster_points

                centroids = kmeans.cluster_centers_

                # Trả về dữ liệu dưới dạng JSON
                return jsonify({
                    "clusters_info": clusters_info,
                    "centroids": centroids.tolist(),  # Chuyển đổi numpy array thành list
                })
                

    return render_template("index.html", image=None)

if __name__ == "__main__":
    app.run(debug=True)
    