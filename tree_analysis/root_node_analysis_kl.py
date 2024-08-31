import xgboost
import json
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import _tree


def process_xgboost_tree(tree_dict, idx, root_node_info, feature_names):
    # 检查是否有分裂点
    if 'split' in tree_dict:
        feature_name = tree_dict['split']
        threshold = tree_dict['split_condition']
        left_values, right_values = None, None

        # 获取子节点信息
        children = tree_dict.get('children', [])
        if len(children) > 0:
            if 'leaf' in children[0]:
                left_values = float(children[0]['leaf'])
            if len(children) > 1 and 'leaf' in children[1]:
                right_values = float(children[1]['leaf'])

        root_node_info.append({
            "Model Type": "XGBoost",
            "Tree Index": idx + 1,
            "Feature": feature_name,
            "Threshold": threshold,
            "Left Child Values": left_values if left_values is not None else "No left child",
            "Right Child Values": right_values if right_values is not None else "No right child",
            "Left Greater Than Right": left_values > right_values if left_values is not None and right_values is not None else "Not comparable"
        })


def collect_tree_data(tree_, idx, root_node_info, feature_names):
    node = 0
    feature_index = tree_.feature[node]
    name = feature_names[feature_index] if feature_index != _tree.TREE_UNDEFINED else "undefined"
    threshold = tree_.threshold[node]
    left_child = tree_.children_left[node]
    right_child = tree_.children_right[node]
    left_values = np.mean(tree_.value[left_child])
    right_values = np.mean(tree_.value[right_child])

    root_node_info.append({
        "Model Type": "Gradient Boosting",
        "Tree Index": idx + 1,
        "Feature": name,
        "Threshold": threshold,
        "Left Child Values": left_values,
        "Right Child Values": right_values,
        "Left Greater Than Right": left_values > right_values
    })


def collect_root_node_info(model_path, feature_names):
    with open(model_path, 'rb') as file:
        model = joblib.load(file)

    root_node_info = []

    if isinstance(model, xgboost.XGBRegressor):
        booster = model.get_booster()
        trees = booster.get_dump(dump_format = 'json')
        for idx, tree_json in enumerate(trees):
            tree_dict = json.loads(tree_json)
            process_xgboost_tree(tree_dict, idx, root_node_info, feature_names)
    elif isinstance(model, GradientBoostingRegressor):
        estimators = model.estimators_.ravel()
        for idx, estimator in enumerate(estimators):
            tree_ = estimator.tree_
            collect_tree_data(tree_, idx, root_node_info, feature_names)

    return root_node_info


def process_models_and_save_to_excel(model_paths, feature_names, file_name):
    all_root_nodes_info = []
    for idx, path in enumerate(model_paths):
        print(f'Processing model {idx + 1} from {path}')
        root_nodes_info = collect_root_node_info(path, feature_names)
        all_root_nodes_info.extend(root_nodes_info)

    df = pd.DataFrame(all_root_nodes_info)
    df.to_excel(file_name, index = False)
    print(f"Data saved to {file_name}")


# 设置模型路径和特征名称
df = pd.read_excel('../Final_Model/Full.xlsx', index_col=0)
feature_names = df.columns[:-4].tolist()


# 模型路径
rf_model_paths = [f"../Final_Model/models/kl/gboost/gboost{i}.pkl" for i in range(1, 6)]
gb_model_paths = [f"../Final_Model/models/kl/xgboost/xgboost{i}.pkl" for i in range(1, 6)]
all_model_paths = rf_model_paths + gb_model_paths

# 处理模型并保存数据
process_models_and_save_to_excel(all_model_paths, feature_names, "kl_Root_Node_Info.xlsx")
