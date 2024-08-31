import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib
import matplotlib.pyplot as plt

# 定义文件夹路径
folder_path_qf = "qf_models"
# 权重定义
weights_qf = {'xgboost': 0.7 / 5, 'rf': 0.3 / 5}

# 加载测试集
df_qf = pd.read_excel('qf_models/train_set_new.xlsx', index_col=0)
X_test_qf = df_qf.drop(columns=['Precipitate Distribution', 'Yield_Strength', 'Tensile_Strength (UTS)', '追踪编号'])
y_test_qf = df_qf['Yield_Strength']
track_numbers = df_qf['追踪编号']

# 选择样本索引
sample_indices = np.arange(X_test_qf.shape[0])

# 特征及其变化范围
feature_ranges = {
    'Calculated Yield Strength': np.arange(50, 301, 1),  # 已知
    'Grain Size': np.arange(5, 31, 0.5),  # 已知
    'Y fraction': np.arange(0.005, 0.04, 0.001),
    'Gd fraction': np.arange(0.005, 0.04, 0.001)
}

# 创建Excel文件以保存结果
with pd.ExcelWriter('qf_multiple_single_variable_analysis.xlsx', engine='openpyxl') as writer:
    for feature, ranges in feature_ranges.items():
        results = []
        for sample_index in sample_indices:
            sample = X_test_qf.iloc[sample_index].copy()
            track_number = track_numbers.iloc[sample_index]
            max_change = 0
            max_change_value = None
            max_change_prediction = None
            previous_prediction = None
            for value in ranges:
                modified_sample = sample.copy()
                modified_sample[feature] = value
                modified_sample_df = pd.DataFrame([modified_sample], columns=X_test_qf.columns)

                prediction_for_this_value = np.zeros(1)
                # 遍历每个模型进行预测
                for model_name, weight in weights_qf.items():
                    for i in range(1, 6):
                        model_path = f'{folder_path_qf}/{model_name}{i}_new.pkl'
                        model = joblib.load(model_path)
                        prediction_for_this_value += model.predict(modified_sample_df) * weight

                if previous_prediction is not None:
                    change = abs(prediction_for_this_value[0] - previous_prediction)
                    if change > max_change:
                        max_change = change
                        max_change_value = value
                        max_change_prediction = prediction_for_this_value[0]

                previous_prediction = prediction_for_this_value[0]

            results.append((track_number, max_change_value, max_change_prediction))

        # 将结果转换为 DataFrame 并保存到 Excel的不同sheet中
        results_df = pd.DataFrame(results, columns=['Track Number', f'{feature} with Max Change', 'Predicted Yield Strength'])
        sheet_name = f'{feature}_max_change'
        results_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(results_df[f'{feature} with Max Change'], bins=20, edgecolor='black')
        plt.xlabel(f'{feature} with Max Change')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {feature} with Max Change')
        plt.grid(True)
        plt.savefig(f'{feature}_max_change_histogram.png')
        plt.close()

print("Multi-feature single variable analysis completed and results saved to Excel and histograms.")


# 抗拉强度结果变化
import numpy as np
import pandas as pd
import joblib

# 定义文件夹路径
folder_path_kl = "kl_models"
# 权重定义
weights_kl = {'xgboost': 0.5 / 5, 'rf': 0.5 / 5}

# 加载测试集
df_kl = pd.read_excel('kl_models/test_set_new.xlsx', index_col=0)
X_test_kl = df_kl.drop(columns=['Precipitate Distribution', 'Yield_Strength', 'Tensile_Strength (UTS)', '追踪编号'])
y_test_kl = df_kl['Tensile_Strength (UTS)']

# 选择一个样本

# 选择样本索引
sample_indices = np.arange(X_test_qf.shape[0])

# 特征及其变化范围
feature_ranges = {
    'Calculated Yield Strength': np.arange(50, 301, 2),  # 已知
    'Grain Size': np.arange(5, 31, 0.5),  # 已知
    'Y fraction': np.arange(0.005, 0.04, 0.001),
    'Gd fraction': np.arange(0.005, 0.04, 0.001)
}

# 创建Excel文件以保存结果
with pd.ExcelWriter('kl_multiple_single_variable_analysis.xlsx', engine='openpyxl') as writer:
    for feature, ranges in feature_ranges.items():
        predictions_changes = []
        for value in ranges:
            modified_sample = sample.copy()
            modified_sample[feature] = value
            modified_sample_df = pd.DataFrame([modified_sample], columns=X_test_kl.columns)

            prediction_for_this_value = np.zeros(1)

            # 遍历每个模型进行预测
            for model_name, weight in weights_kl.items():
                for i in range(1,6):
                    model_path = f'{folder_path_kl}/{model_name}{i}_new.pkl'
                    model = joblib.load(model_path)
                    prediction_for_this_value += model.predict(modified_sample_df) * weight

            predictions_changes.append((value, prediction_for_this_value[0]))

        # 将结果转换为 DataFrame 并保存到 Excel的不同sheet中
        results_df = pd.DataFrame(predictions_changes, columns=[feature, 'Predicted Tensile Strength'])
        results_df.to_excel(writer, sheet_name=feature, index=False)

print("Multi-feature single variable analysis for tensile strength completed and results saved to Excel.")
