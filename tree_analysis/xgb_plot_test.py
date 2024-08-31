import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
import joblib
import pandas as pd
# 假设 `bst` 是你的训练好的 XGBoost 模型
# 如果你需要从文件加载模型，可以使用：
# bst = xgb.Booster()
# bst.load_model('model_path')
df = pd.read_excel('../Final_Model/Full.xlsx', index_col=0)
feature_names = df.columns.tolist()

# 模型路径列表
model_paths = [f"../Final_Model/models/kl/xgboost/xgboost{i}.pkl" for i in range(1, 6)]

bst = joblib.load(model_paths[2])

# 绘制第一棵树
plt.figure(figsize=(20, 16),dpi=1000)
plot_tree(bst, num_trees=0)

plt.savefig('kl_xgboost1_vector.svg')  # 保存为 SVG 格式
plt.show()
