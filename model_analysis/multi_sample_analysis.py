import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

# 通用函数：单变量分析并保存结果
def analyze_and_save_results(df, X_test, y_test, folder_path, weights, output_file, target_name, target_col_name, feature_ranges):
    sample_indices = range(0,X_test.shape[0],2)
    if not os.path.exists(output_file):
        df = pd.DataFrame().to_excel(output_file)
    else:
        print(f'{output_file}文件已存在')
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for idx, (feature, ranges) in enumerate(feature_ranges.items(), 1):
            print(f"Processing feature '{feature}' ({idx}/{len(feature_ranges)})...")
            results = []
            for sample_index in sample_indices:
                # if sample_index//50==0:
                #     print(f'处理sample{sample_index//50}-{sample_index//50+50}个样本')
                sample = X_test.iloc[sample_index].copy()
                track_number = df['追踪编号'].iloc[sample_index]
                max_change = 0
                max_change_value = None
                max_change_prediction = None
                previous_prediction = None
                for value in ranges:
                    modified_sample = sample.copy()
                    modified_sample[feature] = value
                    modified_sample_df = pd.DataFrame([modified_sample], columns=X_test.columns)

                    prediction_for_this_value = np.zeros(1)
                    for model_name, weight in weights.items():
                        for i in range(0, 5):
                            model_path = f'{folder_path}/{model_name}_{i}.pkl'
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
            print(f"Processing feature '{feature}' completed.")

            results_df = pd.DataFrame(results, columns=['Track Number', f'{feature} with Max Change', f'Predicted {target_name}'])
            sheet_name = f'{feature}_max_change_{target_name}'
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)

            plt.figure(figsize=(10, 6))
            plt(results_df[f'{feature} with Max Change'], bins=20, edgecolor='black')
            plt.xlabel(f'{feature} with Max Change')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {feature} with Max Change ({target_name})')
            plt.grid(True)
            plt.savefig(f'histogram_results/{feature}_max_change_histogram_{target_name}.png')
            plt.close()

    print(f"Multi-feature single variable analysis for {target_name} completed and results saved to Excel and histograms.")

# 加载数据
df_qf = pd.read_excel('20240607FULL_final.xlsx', index_col=0, sheet_name='qf_21')
X_test_qf = df_qf.iloc[:, :-3]
y_test_qf = df_qf['Yield_Strength']

df_kl = pd.read_excel('20240607FULL_final.xlsx', index_col=0, sheet_name='kl_21')
X_test_kl = df_kl.iloc[:, :-3]
y_test_kl = df_kl['Tensile_Strength （UTS）']

# 定义模型路径和权重
folder_path_qf = "small_model/final_model/qf/rf"
weights_qf = {'rf': 0.2}
folder_path_kl = "small_model/final_model/kl/rf"
weights_kl = {'rf': 0.2}

# 小模型特征及其变化范围
small_model_feature_ranges = {
    'Diameter of basal phase': np.arange(1, 100, 1),
    'Width of basal phase': np.arange(0.5, 20, 0.2),
    'Diameter of prismatic phase': np.arange(0, 31, 0.5),
}

# 分析并保存屈服强度和抗拉强度的结果（小模型）
analyze_and_save_results(df_qf, X_test_qf, y_test_qf, folder_path_qf, weights_qf, 'small_qf_multiple_single_variable_analysis.xlsx', 'Yield Strength', 'Yield_Strength', small_model_feature_ranges)
analyze_and_save_results(df_kl, X_test_kl, y_test_kl, folder_path_kl, weights_kl, 'small_kl_multiple_single_variable_analysis.xlsx', 'Tensile Strength', 'Tensile_Strength （UTS）', small_model_feature_ranges)

# 通用模型数据加载
df_qf_generic = pd.read_excel('general_model/qf_models/train_set_new.xlsx', index_col=0)
df_kl_generic = pd.read_excel('general_model/kl_models/train_set_new.xlsx', index_col=0)


track_numbers_qf_generic = df_qf_generic['追踪编号']
X_test_qf_generic = df_qf_generic.drop(columns=['Precipitate Distribution', 'Yield_Strength', 'Tensile_Strength (UTS)', '追踪编号'])
y_test_qf_generic = df_qf_generic['Yield_Strength']

track_numbers_kl_generic = df_kl_generic['追踪编号']
X_test_kl_generic = df_kl_generic.drop(columns=['Precipitate Distribution', 'Yield_Strength', 'Tensile_Strength (UTS)', '追踪编号'])
y_test_kl_generic = df_kl_generic['Tensile_Strength (UTS)']

# 定义模型路径和权重（通用模型）
folder_path_qf_generic = "general_model/qf_models"
folder_path_kl_generic = "general_model/kl_models"
weights_qf_generic = {'xgboost': 0.7 / 5, 'rf': 0.3 / 5}
weights_kl_generic = {'xgboost': 0.5 / 5, 'rf': 0.5 / 5}

# 通用模型特征及其变化范围
generic_model_feature_ranges = {
    'Calculated Yield Strength': np.arange(50, 301, 5),
    'Grain Size': np.arange(5, 31, 1),
    'Y fraction': np.arange(0.005, 0.04, 0.002),
    'Gd fraction': np.arange(0.005, 0.04, 0.002)
}

# 分析并保存屈服强度和抗拉强度的结果（通用模型）
analyze_and_save_results(df_qf_generic, X_test_qf_generic, y_test_qf_generic, folder_path_qf_generic, weights_qf_generic, 'generic_qf_multiple_single_variable_analysis.xlsx', 'Yield Strength', 'Yield_Strength', generic_model_feature_ranges)
analyze_and_save_results(df_kl_generic, X_test_kl_generic, y_test_kl_generic, folder_path_kl_generic, weights_kl_generic,
                         'generic_kl_multiple_single_variable_analysis_train_set.xlsx', 'Tensile Strength', 'Tensile_Strength (UTS)', generic_model_feature_ranges)

print("Multi-feature single variable analysis for both models completed and results saved to Excel and histograms.")
