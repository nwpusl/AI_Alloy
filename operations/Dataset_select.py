import os.path
import pandas as pd
file_count = 5
# 定义一个列名从中文到英文的映射字典
columns_translation = {
    '预测屈服强度': 'qf',
    '预测抗拉强度 （UTS）': 'kl'
}
def filter_samples(file_path, threshold_dict, output_file):
    data = pd.read_excel(file_path)
    # 重命名列为英文
    data.rename(columns=columns_translation, inplace=True)
    filtered_samples = []  # 保存符合条件的样本列表
    for index, row in data.iterrows():
        is_filtered = True
        for key, value in threshold_dict.items():
            # 确保该行中的值对应于有效键，以及比较操作不是在None和数值类型间进行
            if pd.isnull(row.get(key)) or row.get(key) <= value:
                is_filtered = False
                break
        if is_filtered:
            filtered_samples.append(row)
    filtered_data = pd.DataFrame(filtered_samples)
    filtered_data.to_excel(output_file, index=False)
    print(f"筛选后的数据已保存至：{output_file}")
    return len(filtered_data)
import os
def filter_datasets_and_save(predicted_yield_strength_threshold, predicted_tensile_strength_threshold):
    # 定义列名映射字典
    columns_translation = {
        '预测屈服强度': 'qf',
        '预测抗拉强度 （UTS）': 'kl'
    }
    # 阈值字典
    threshold_dict = {
        'qf': predicted_yield_strength_threshold,
        'kl': predicted_tensile_strength_threshold
    }
    # 输出文件夹名称
    threshold_string = '_'.join([f"{key}-{value}" for key, value in threshold_dict.items()])
    output_dir = f'filtered_morphology/{threshold_string}'
    os.makedirs(output_dir, exist_ok=True)
    # 文件处理
    total_filtered_count = 0
    for i in range(1, file_count):
        input_file_path = f'morphology_decoding/alloy_dataset{i}.xlsx'
        output_file_path = f'{output_dir}/filtered_morphology{i}.xlsx'
        # 数据读取与列名重命名
        data = pd.read_excel(input_file_path).rename(columns=columns_translation)
        filtered_samples = []
        # 数据过滤
        for _, row in data.iterrows():
            if all(row.get(key, 0) > value for key, value in threshold_dict.items()):
                filtered_samples.append(row)
        # 保存过滤后的数据
        filtered_data = pd.DataFrame(filtered_samples)
        filtered_data.to_excel(output_file_path, index=False)
        # 更新计数
        total_filtered_count += len(filtered_samples)
        print(f'文件 {input_file_path} 筛选出 {len(filtered_samples)} 个符合条件的样本，已保存至 {output_file_path}')
    print(f'总共筛选出 {total_filtered_count} 个样本')
# 使用函数
filter_datasets_and_save(350, 400)
filter_datasets_and_save(360, 400)
filter_datasets_and_save(360, 380)
filter_datasets_and_save(380, 420)
filter_datasets_and_save(370, 420)
filter_datasets_and_save(375, 420)
filter_datasets_and_save(380, 400)
filter_datasets_and_save(390, 420)