import os
import pickle
import pandas as pd
import numpy as np

def get_top_features(model, top_n=15):
    """获取模型前N个最重要的特征及其重要性"""
    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()[::-1][:top_n]
    top_features = [(f"feature_{idx+1}", importance) for idx, importance in enumerate(feature_importances[sorted_idx], start=1)]
    return top_features

def analyze_and_aggregate_fully(root_dir, output_excel_name):
    writer = pd.ExcelWriter(output_excel_name, engine='xlsxwriter')
    problem_folders = ['kl', 'qf']

    for problem_folder in problem_folders:
        all_importances = []
        weighted_importances = []
        total_weights = {'rf': 0.3 / 5, 'rf': 0.3 / 5, 'xgboost': 0.4 / 5}

        # 对每个问题分别处理
        problem_dir = os.path.join(root_dir, problem_folder)
        models_count = sum([len(files) for r, d, files in os.walk(problem_dir)])
        for model_type in os.listdir(problem_dir):
            model_type_dir = os.path.join(problem_dir, model_type)
            for model_file in os.listdir(model_type_dir):
                model_path = os.path.join(model_type_dir, model_file)
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                # 单个模型特征重要性
                top_features = get_top_features(model)
                df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
                sheet_name = f"{model_type}_{model_file.split('.')[0]}_importance"
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

                # 收集为平均和加权平均
                importances = model.feature_importances_
                if problem_folder == 'kl':  # 仅kl问题有不同权重
                    weight = total_weights.get(model_type, 1 / models_count)
                else:
                    weight = 1 / models_count  # qf或其他问题均等权重

                if len(all_importances) < 1:
                    all_importances = np.zeros_like(importances)
                    weighted_importances = np.zeros_like(importances)

                all_importances += importances
                weighted_importances += importances * weight

        # 计算和保存平均特征重要性
        avg_importances = all_importances / models_count
        top_avg_features = get_top_n_features(avg_importances, 15)
        df_avg = pd.DataFrame(top_avg_features, columns=['Feature', 'Importance'])
        df_avg.to_excel(writer, sheet_name=f"{problem_folder}_avg_importance"[:31], index=False)

        # 计算和保存加权平均特征重要性
        weighted_avg_importances = weighted_importances / np.sum(list(total_weights.values()) if problem_folder == 'kl' else [1/models_count])
        top_weighted_features = get_top_n_features(weighted_avg_importances, 15)
        df_weighted = pd.DataFrame(top_weighted_features, columns=['Feature', 'Importance'])
        df_weighted.to_excel(writer, sheet_name=f"{problem_folder}_weighted_importance"[:31], index=False)

    writer.save()

# Utility function to obtain top n features from importances
def get_top_n_features(importances, top_n):
    sorted_indices = np.argsort(importances)[::-1][:top_n]
    top_features = [(f"feature_{i+1}", importances[i]) for i in sorted_indices]
    return top_features

# 调用函数进行全面分析和汇总
root_dir = 'models'  # 替换为您的模型文件夹路径
analyze_and_aggregate_fully(root_dir, 'kl_and_qf_analysis.xlsx')  # 生成包含kl和qf分析结果的Excel文件