import os.path
import os
import time
import joblib
import numpy as np
import pandas as pd
weights_qf = {'rf': 0.5/5, 'rf': 0.5/5}  # 注意所有模型系数和为1
weights_kl = {'rf': 0.3/5, 'rf': 0.3/5, 'xgboost': 0.4/5}
new_column_names = {'qf': '预测屈服强度', 'kl': '预测抗拉强度 （UTS）'}
# 预测函数, 这里需要遍历所有的文件
model_num_per_algorithm = 5
def predict(data, models_folder, weights):
    predictions = []
    for model_name, weight in weights.items():
        for i in range(1, model_num_per_algorithm + 1):
            model_path = f'models/{models_folder}/{model_name}/{model_name}{i}.pkl'
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
        print(f'正在处理前{(index-1)*30}-{index*30}w条数据')
        data_path = os.path.join(folder_path, dataset)
        data = pd.read_excel(data_path, index_col=0)
        # 需要使用predict函数的data列名根据实际情况进行筛选
        prediction_qf = predict(data, 'qf', weights_qf)
        prediction_kl = predict(data, 'kl', weights_kl)
        # 将预测结果添加到表格末尾
        data[new_column_names['qf']] = prediction_qf
        data[new_column_names['kl']] = prediction_kl
        save_dir='alloy_dataset_prediction_results_expanded'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        result_path=save_dir+f'/Alloy_dataset_results{index}.xlsx'
        # 保存带有预测结果的新表格
        data.to_excel(result_path, index=False)
        toc=time.time()
        print(f"Processed and saved: {result_path}")
        print(f"前{(index-1)*30}-{index*30}w条数据花费时间: ",toc-tic,'s')
# 设置文件夹路径并处理所有数据集
folder_path = 'alloy_dataset_results_expanded'
process_datasets(folder_path)
# 将最终预测结果返回最初的表格
# 1-6根据情况增加或者减少
folder_path = 'alloy_dataset_prediction_results_expanded'
# 获取指定文件夹下文件的数量
file_count = sum(1 for entry in os.scandir(folder_path) if entry.is_file())
for i in range(1, file_count+1):
    # 处理第i个列表
    tic=time.time()
    # 读取combinations文件
    combinations_df = pd.read_excel(f'original_mass_results_expanded/origin_Alloy_dataset{i}.xlsx')
    # 读取对应的Alloy_dataset文件
    alloy_dataset_df = pd.read_excel(f'alloy_dataset_prediction_results_expanded/Alloy_dataset_results{i}.xlsx')
    # 提取指定特征列
    selected_features = alloy_dataset_df[['晶粒尺寸', '惯习面', '析出相分布', '计算固溶', '计算晶界', '计算总强度', '预测屈服强度', '预测抗拉强度 （UTS）']]
    # 索引重置
    combinations_df.reset_index(drop=True, inplace=True)
    selected_features.reset_index(drop=True, inplace=True)
    # 合并combinations和Alloy_dataset文件的数据
    merged_data = pd.concat([combinations_df, selected_features], axis=1)
    merged_data.insert(0, 'Index', range(1, len(merged_data) + 1))
    save_dir='elements_and_targets'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # 保存合并后的数据到新的Excel文件
    merged_data.to_excel(f'elements_and_targets/merged_data_{i}.xlsx', index=False)
    toc=time.time()
    print(f'第{i}组文件处理花费{toc-tic}s')
# 将形貌编码还原
Grain_Size = {2: '小尺寸', 10: '中位数', 42: '大尺寸'}
Habit_Panel = {0: '基面', 1: '基面+杂质', 2: '基面+杂质+柱面', 3: '基面+柱面', 4: '无', 5: '杂质', 6: '杂质+柱面',7: '柱面'}
Precipitation_phase_distribution = {0: '无', 1: '晶内', 2: '晶界', 3: '晶界+晶内', 4: '晶粒'}
save_dir='morphology_decoding'
if not os.path.exists(save_dir):
    os.makedirs(save_dir,exist_ok=True)
for i in range(1,file_count+1):
    print(f'正在转换第{i}个表格')
    tic=time.time()
    # 读取合并后的数据
    merged_data = pd.read_excel(f'elements_and_targets/merged_data_{i}.xlsx',index_col=0)
    # 将编码数字转换为原始键值
    merged_data['晶粒尺寸'] = merged_data['晶粒尺寸'].map(Grain_Size)
    merged_data['惯习面'] = merged_data['惯习面'].map(Habit_Panel)
    merged_data['析出相分布'] = merged_data['析出相分布'].map(Precipitation_phase_distribution)
    # 保存转换后的数据到新的Excel文件
    merged_data.to_excel(save_dir+f'/alloy_dataset{i}.xlsx', index=False)
    toc=time.time()
    print(f'第{i}个表格编码转换用时{toc-tic}')
