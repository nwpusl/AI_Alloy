

import pandas as pd
import os


def merge_excel_sheets(input_folder, output_file):
    writer = pd.ExcelWriter(output_file, engine = 'openpyxl')

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if os.path.isfile(file_path) and filename.endswith('.xlsx'):
            try:
                xls = pd.ExcelFile(file_path, engine = 'openpyxl')

                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name = sheet_name)

                    # 使用sheet名的最后一个字符和文件名的前部分创建新的sheet名
                    last_char = sheet_name[-1] if len(sheet_name) > 0 else 'Empty'
                    new_sheet_name = f"{filename[:10]}_{last_char}"
                    new_sheet_name = new_sheet_name[:31]  # 确保sheet名长度不超过31个字符

                    df.to_excel(writer, sheet_name = new_sheet_name, index = False)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

    writer.save()
# 使用函数
input_folder = 'all_performance_excel'  # 替换为包含Excel文件的文件夹的路径
output_file = 'all_performance_excel/mergerd_performance_excel.xlsx'  # 输出文件的名称和路径
merge_excel_sheets(input_folder, output_file)
