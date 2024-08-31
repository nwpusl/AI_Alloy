# 这里针对的是通用模型

import numpy as np
import joblib
import pandas as pd

from sklearn.tree import _tree

def collect_root_node_info(model_path, feature_names):
    # 加载模型
    with open(model_path, 'rb') as file:
        model = joblib.load(file)

    root_node_info = []

    # 确定是否为梯度提升并正确处理
    estimators = model.estimators_
    if isinstance(estimators, np.ndarray):
        if estimators.ndim == 2:  # 对于梯度提升，需要展平数组
            estimators = estimators.ravel()

    # 遍历模型中的每棵树
    for idx, tree in enumerate(estimators):
        tree_ = tree.tree_ if hasattr(tree, 'tree_') else tree

        node = 0
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            continue  # 跳过未定义特征的节点

        feature_index = tree_.feature[node]
        name = feature_names[feature_index] if feature_index != -2 else "undefined"
        threshold = tree_.threshold[node]
        left_child = tree_.children_left[node]
        right_child = tree_.children_right[node]
        left_values = np.mean(tree_.value[left_child])
        right_values = np.mean(tree_.value[right_child])

        root_node_info.append({
            "Model Type": "Random Forest" if 'rf' in model_path else "Gradient Boosting",
            "Tree Index": idx + 1,
            "Feature": name,
            "Threshold": threshold,
            "Left Child Values": left_values,
            "Right Child Values": right_values,
            "Left Greater Than Right": left_values > right_values
        })

    return root_node_info


def process_models_and_save_to_excel(model_paths, feature_names, file_name):
    all_root_nodes_info = []

    # 遍历所有模型路径，收集根节点信息
    for idx, path in enumerate(model_paths):
        print(f'Processing model {idx + 1} from {path}')
        root_nodes_info = collect_root_node_info(path, feature_names)
        all_root_nodes_info.extend(root_nodes_info)

    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(all_root_nodes_info)
    df.to_excel(file_name, index=False)
    print(f"Data saved to {file_name}")

# 确保特征名称列表是从正确的DataFrame列获取
df = pd.read_excel('../Final_Model/Full.xlsx', index_col=0)
df = df[df['Yield_Strength'] != 0].reset_index(drop=True)
feature_names = df.columns[:-4].tolist()

# 模型路径
rf_model_paths = [f"../Final_Model/models/qf/rf/rf{i}.pkl" for i in range(1, 6)]
gb_model_paths = [f"../Final_Model/models/qf/gboost/gboost{i}.pkl" for i in range(1, 6)]
all_model_paths = rf_model_paths + gb_model_paths

# 处理模型并保存数据
process_models_and_save_to_excel(all_model_paths, feature_names, "qf_Root_Node_Info.xlsx")
