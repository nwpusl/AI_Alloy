import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_validate,cross_val_score,KFold
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from utils import create_excel
qf_data = pd.read_excel('../20240607Feature.xlsx', index_col = 0)

X = qf_data.drop(['Yield_Strength', 'Tensile_Strength （UTS）','追踪编号'],axis = 1)
print('总共有{}个特征'.format(X.shape[1]))
y = qf_data['Yield_Strength']
print(list(y))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
bins = np.linspace(y.min(), y.max(), 11)
y_binned = np.digitize(y, bins)
n_splits = 5
# 初始化StratifiedKFold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=200)

# 初始化随机森林回归器，最大深度为40，预测器（树）数量为100
import xgboost as xgb
# 初始化列表以存储每个折叠的分数
r2_scores = []
mape_scores = []
mae_scores= []
feature_importances = np.zeros(X.shape[1])

index = 0
for train_index,test_index in skf.split(X,y_binned):
    # print(train_index,test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 初始化 XGBoost 回归器
    model = RandomForestRegressor(random_state = 60)

    # 训练模型
    model.fit(X_train, y_train)
    feature_importances += model.feature_importances_

    # 预测测试集
    y_pred = model.predict(X_test)
    print(type(y_test))
    # print(len(y_pred))
    # print(len(y_test))
    # print(y_test)
    for i in range(len(y_test)):
        print(np.abs(y_pred[i]-y_test.iloc[i])) # 对series格式的话采用iloc进行行索引

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test,y_pred)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    mae_scores.append(mae)
    mape_scores.append(mape)
print(f"Mean Absolute Percentage Error:{np.mean(mape_scores)}")
print(f"Mean Absolute Error: {np.mean(mae_scores)}")
print(f"R^2 Score: {np.mean(r2_scores)}")

average_feature_importances = feature_importances / n_splits

# 将特征重要性排序
sorted_indices = np.argsort(average_feature_importances)[::-1]

# 输出到新的表格中
importance_output_file = 'average_feature_importance_qf.xlsx'
create_excel(importance_output_file)
feature_names = X.columns[sorted_indices]
feature_importance_values = average_feature_importances[sorted_indices]
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_values})
with pd.ExcelWriter(importance_output_file,engine = 'openpyxl',mode='a',if_sheet_exists = 'replace') as writer:
    feature_importance_df.to_excel(writer, sheet_name='qf_289features',index=False)
    print(f'文件保存到{importance_output_file}中')

scores_file= 'scores_file.xlsx'
create_excel(scores_file)
with pd.ExcelWriter(scores_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    scores_data = {'MAE': [np.mean(mae_scores)], 'MAPE': [np.mean(mape_scores)], 'R^2': [np.mean(r2_scores)]}
    df = pd.DataFrame(scores_data)
    df.to_excel(writer, sheet_name='qf_289eatures', index=False)
print(f'分数文件已保存到{scores_file}文件中')

