import pandas as pd
import numpy as np
import joblib
import os
import time
# 定义权重和列名
weights_qf = {'rf': 0.5/5, 'rf': 0.5/5}  # 注意所有模型系数和为1
weights_kl = {'rf': 0.3/5, 'rf': 0.3/5, 'xgboost': 0.4/5}
new_column_names = {'qf': '预测屈服强度', 'kl': '预测抗拉强度 （UTS）'}

# 预测函数
def predict(data, models_folder, weights):
    predictions = []
    for model_name, weight in weights.items():
        for i in range(1, 6):
            model_path = f'{models_folder}/{model_name}/{model_name}{i}.pkl'
            model = joblib.load(model_path)
            prediction = model.predict(data) * weight
            predictions.append(prediction)

    final_prediction = np.sum(predictions, axis=0)
    return final_prediction

# 主处理函数
def process_datasets(folder_path):
    datasets = [f for f in os.listdir(folder_path) if f.startswith('Alloy_dataset') and f.endswith('.xlsx')]
    index=0
    for dataset in datasets:
        tic = time.time()
        index=index+1
        print(f'正在处理前{index*40}-{(index+1)*40}w条数据')
        data_path = os.path.join(folder_path, dataset)
        data = pd.read_excel(data_path, index_col=0)
        # 下面两行代码test用
        test_num=1
        data=data.iloc[0:test_num+1,:]
        #
        # 需要使用predict函数的data列名根据实际情况进行筛选
        prediction_qf = predict(data, 'qf', weights_qf)
        prediction_kl = predict(data, 'kl', weights_kl)

        # 将预测结果添加到表格末尾
        data[new_column_names['qf']] = prediction_qf
        data[new_column_names['kl']] = prediction_kl

        result_path=f'alloy_dataset_prediction_results/Alloy_dataset_results{index}.xlsx'
        # 保存带有预测结果的新表格
        data.to_excel(result_path, index=False)
        toc=time.time()
        print(f"Processed and saved: {result_path}")
        print(f"前{index*40}-{(index+1)*40}w条数据花费时间: ",toc-tic,'s')

# 设置文件夹路径并处理所有数据集
folder_path = 'alloy_dataset_results_expanded'
if not os.path.exists('alloy_dataset_prediction_results'):
    os.mkdir('alloy_dataset_prediction_results')
process_datasets(folder_path)