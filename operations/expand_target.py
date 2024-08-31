import pandas as pd
import os
def count_excel_files(folder_path):
    # 确保文件夹路径存在
    if not os.path.isdir(folder_path):
        print(f"错误：路径 {folder_path} 不存在！")
        return
    # 列出文件夹下所有文件
    files = os.listdir(folder_path)
    # 统计.xlsx文件数量
    excel_files = [file for file in files if file.endswith('.xlsx')]
    return len(excel_files)
folder_path = 'morphology_decoding'
file_count = count_excel_files(folder_path)
# 定义一个列名从中文到英文的映射字典
columns_translation = {
    '预测屈服强度': 'qf',
    '预测抗拉强度 （UTS）': 'kl'
}
# 元素列表
elements_to_keep = ['Zn', 'Gd', 'Y', 'Zr']
def filter_samples(file_path, threshold_dict, output_file):
    data = pd.read_excel(file_path)
    filtered_samples = []  # 保存符合条件的样本列表
    for index, row in data.iterrows():
        # 检查元素列是否包含所需元素
        present_elements = [element for element in elements_to_keep if row.get(element, 0) > 0]
        if len(present_elements) >= 2:  # 至少两种选择的元素出现
            is_filtered = True
            # 检查阈值条件
            for key, value in threshold_dict.items():
                if pd.isnull(row.get(key)) or row.get(key) <= value:
                    is_filtered = False
                    break
            if is_filtered:
                filtered_samples.append(row)
    filtered_data = pd.DataFrame(filtered_samples)
    filtered_data.to_excel(output_file, index=False)
    print(f"筛选后的数据已保存至：{output_file}")
    return len(filtered_data)
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
    output_dir = f'filtered_morphology/target_system/{threshold_string}'
    os.makedirs(output_dir, exist_ok=True)
    # 文件处理
    total_filtered_count = 0
    for i in range(1, file_count + 1):
        input_file_path = f'morphology_decoding/alloy_dataset{i}.xlsx'
        output_file_path = f'{output_dir}/filtered_morphology{i}.xlsx'
        # 数据读取
        data = pd.read_excel(input_file_path)
        # 过滤数据
        filtered_samples = []
        for _, row in data.iterrows():
            # 检查元素列是否包含所需元素
            present_elements = [element for element in elements_to_keep if row.get(element, 0) > 0]
            if len(present_elements) >= 2:  # 至少两种选择的元素出现
                if all(row.get(key, 0) > value for key, value in threshold_dict.items()):
                    filtered_samples.append(row)
        # 保存过滤后的数据
        filtered_data = pd.DataFrame(filtered_samples)
        filtered_data.rename(columns=columns_translation, inplace=True)  # 重命名列为英文
        filtered_data.to_excel(output_file_path, index=False)
        # 更新计数
        total_filtered_count += len(filtered_samples)
        print(f'文件 {input_file_path} 筛选出 {len(filtered_samples)} 个符合条件的样本，已保存至 {output_file_path}')
    print(f'总共筛选出 {total_filtered_count} 个样本')
# 调用函数并保留 threshold_dict
filter_datasets_and_save(300, 300)
