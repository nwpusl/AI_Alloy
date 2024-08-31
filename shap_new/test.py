# 获取shap值
import pandas as pd


def remove_importance_sheets(file_path, output_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)

    # Initialize a writer for the new file
    with pd.ExcelWriter(output_path) as writer:
        # Iterate over all sheet names
        for sheet_name in xls.sheet_names:
            # Check if the sheet name ends with '_importance'
            if not sheet_name.endswith('_importance'):
                # Read the sheet
                df = pd.read_excel(xls, sheet_name = sheet_name)
                # Write the sheet to the new file
                df.to_excel(writer, sheet_name = sheet_name, index = False)


# Define file paths
qf_input_path = 'index/qf_index.xlsx'
kl_input_path = 'index/kl_index.xlsx'
qf_output_path = 'index/qf_shap.xlsx'
kl_output_path = 'index/kl_shap.xlsx'

# Remove _importance sheets and save new files
remove_importance_sheets(qf_input_path, qf_output_path)
remove_importance_sheets(kl_input_path, kl_output_path)
