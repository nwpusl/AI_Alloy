import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_absolute_percentage_error
score_path_qf='predict_performance_qf.xlsx'
score_path_kl='predict_performance_qf.xlsx'

if os.path.exists(score_path_qf):
    print(f"The file '{score_path_qf}' exists.")
else:
    print(f"The file '{score_path_qf}' does not exist.")
    df = pd.DataFrame(columns=['Model', 'R^2', 'MAPE', 'MAE'])
    df.to_excel(score_path_qf, index=False)
if os.path.exists(score_path_kl):
    print(f"The file '{score_path_kl}' exists.")
else:
    print(f"The file '{score_path_kl}' does not exist.")
    df = pd.DataFrame(columns=['Model', 'R^2', 'MAPE', 'MAE'])
    df.to_excel(score_path_kl, index=False)

df = pd.read_excel('sample_validation_data.xlsx')
df = df[df['Yield_Strength'] != 0].reset_index(drop=True)
df = df[df['抗拉强度_backup （UTS）'] != 0].reset_index(drop=True)
# df = df[df['计算晶界'] != 0].reset_index(drop=True)
X_qf = df.iloc[:, :-4].to_numpy() # 模型的输入是mX384
X_kl = df.iloc[:,:-3].to_numpy()
# print(X)
y_qf = df.iloc[:, -4].to_numpy()
y_kl = df.iloc[:,-3].to_numpy()
# print(y)

# weights = {'rf_best.pkl': 0.35, 'xgboost_best_new.pkl': 0.35, 'adaboost_best.pkl': 0.2, 'svm_rbf_best.pkl': 0.1}
# 绝对路径列表
model_paths_kl = [
    'models/kl/qf/rf/gboost_kl.pkl',
    'models/kl/qf/rf/rf_best.pkl',
    'models/kl/qf/xgboost/xgboost_best_new.pkl',
    'models/kl/qf/knn/knn_best.pkl',
    'models/kl/qf/adaboost/adaboost_best.pkl',
    'models/kl/qf/dt/dt_best.pkl',
    'models/kl/qf/ridge/ridge_best.pkl',
    'models/kl/qf/elasticNet/elasticNet_best.pkl',
    'models/kl/qf/lasso/lasso_best.pkl',
    'models/kl/qf/svm_linear/svm_linear_best.pkl',
    'models/kl/qf/svm_rbf/svm_rbf_best.pkl'
]# 假设的模型权重
weights1 = {'gboost_kl.pkl': 1,'rf_best.pkl': 0,'xgboost_best_new.pkl': 0,'knn_best.pkl': 0,'adaboost_best.pkl': 0,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weights2 = {'gboost_kl.pkl': 0,'rf_best.pkl': 1,'xgboost_best_new.pkl': 0,'knn_best.pkl': 0,'adaboost_best.pkl': 0,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weights3 = {'gboost_kl.pkl': 0,'rf_best.pkl': 0,'xgboost_best_new.pkl': 1,'knn_best.pkl': 0,'adaboost_best.pkl': 0,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weights4 = {'gboost_kl.pkl': 0,'rf_best.pkl': 0,'xgboost_best_new.pkl': 0,'knn_best.pkl': 1,'adaboost_best.pkl': 0,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weights5 = {'gboost_kl.pkl': 0,'rf_best.pkl': 0,'xgboost_best_new.pkl': 0,'knn_best.pkl': 0,'adaboost_best.pkl': 1,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weights6 = {'gboost_kl.pkl': 0.22,'rf_best.pkl': 0.22,'xgboost_best_new.pkl': 0.21,'knn_best.pkl': 0.18,'adaboost_best.pkl': 0.17,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weights7 = {'gboost_kl.pkl': 0.2,'rf_best.pkl': 0.2,'xgboost_best_new.pkl': 0.2,'knn_best.pkl': 0.2,'adaboost_best.pkl': 0.2,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weights8 = {'gboost_kl.pkl': 0.5,'rf_best.pkl': 0.5,'xgboost_best_new.pkl': 0,'knn_best.pkl': 0,'adaboost_best.pkl': 0,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weights9 = {'gboost_kl.pkl': 0.4,'rf_best.pkl': 0.3,'xgboost_best_new.pkl': 0.3,'knn_best.pkl': 0,'adaboost_best.pkl': 0,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weights10 = {'gboost_kl.pkl': 0.5,'rf_best.pkl': 0.3,'xgboost_best_new.pkl': 0.2,'knn_best.pkl': 0,'adaboost_best.pkl': 0,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weights11 = {'gboost_kl.pkl': 0.4,'rf_best.pkl': 0.3,'xgboost_best_new.pkl': 0.2,'knn_best.pkl': 0.1,'adaboost_best.pkl': 0,'dt_best.pkl': 0,'ridge_best.pkl': 0, 'elasticNet_best.pkl': 0, 'lasso_best.pkl': 0, 'svm_linear_best.pkl': 0,'svm_rbf_best.pkl': 0}
weight_ls=[weights1,weights2,weights3,weights4,weights5,weights6,weights7,weights8,weights9,weights10,weights11]

model_paths = ['models/qf/rf/gboost_kl.pkl','models/qf/rf/rf_best.pkl','models/qf/xgboost/xgboost_best_new.pkl','models/qf/knn/knn_best.pkl','models/qf/adaboost/adaboost_best.pkl','models/qf/dt/dt_best.pkl','models/qf/ridge/ridge_best.pkl','models/qf/elasticNet/elasticNet_best.pkl', 'models/qf/lasso/lasso_best.pkl', 'models/qf/svm_linear/svm_linear_best.pkl', 'models/qf/svm_rbf/svm_rbf_best.pkl']
weights1_kl = {'gboost_kl.pkl': 1,'rf_kl.pkl': 0,'xgboost_kl.pkl': 0,'knn_kl.pkl': 0,'adaboost_kl.pkl': 0,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weights2_kl = {'gboost_kl.pkl': 0,'rf_kl.pkl': 1,'xgboost_kl.pkl': 0,'knn_kl.pkl': 0,'adaboost_kl.pkl': 0,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weights3_kl = {'gboost_kl.pkl': 0,'rf_kl.pkl': 0,'xgboost_kl.pkl': 1,'knn_kl.pkl': 0,'adaboost_kl.pkl': 0,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weights4_kl = {'gboost_kl.pkl': 0,'rf_kl.pkl': 0,'xgboost_kl.pkl': 0,'knn_kl.pkl': 1,'adaboost_kl.pkl': 0,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weights5_kl = {'gboost_kl.pkl': 0,'rf_kl.pkl': 0,'xgboost_kl.pkl': 0,'knn_kl.pkl': 0,'adaboost_kl.pkl': 1,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weights6_kl = {'gboost_kl.pkl': 0.22,'rf_kl.pkl': 0.22,'xgboost_kl.pkl': 0.21,'knn_kl.pkl': 0.18,'adaboost_kl.pkl': 0.17,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weights7_kl = {'gboost_kl.pkl': 0.2,'rf_kl.pkl': 0.2,'xgboost_kl.pkl': 0.2,'knn_kl.pkl': 0.2,'adaboost_kl.pkl': 0.2,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weights8_kl = {'gboost_kl.pkl': 0.5,'rf_kl.pkl': 0.5,'xgboost_kl.pkl': 0,'knn_kl.pkl': 0,'adaboost_kl.pkl': 0,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weights9_kl = {'gboost_kl.pkl': 0.3,'rf_kl.pkl': 0.3,'xgboost_kl.pkl': 0.4,'knn_kl.pkl': 0,'adaboost_kl.pkl': 0,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl':0}
weights10_kl = {'gboost_kl.pkl': 0.5,'rf_kl.pkl': 0.3,'xgboost_kl.pkl': 0.2,'knn_kl.pkl': 0,'adaboost_kl.pkl': 0,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weights11_kl = {'gboost_kl.pkl': 0.2,'rf_kl.pkl': 0.3,'xgboost_kl.pkl': 0.4,'knn_kl.pkl': 0.1,'adaboost_kl.pkl': 0,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weight12_kl={'gboost_kl.pkl': 0.4,'rf_kl.pkl': 0,'xgboost_kl.pkl': 0.6,'knn_kl.pkl': 0,'adaboost_kl.pkl': 0,'dt_kl.pkl': 0,'ridge_kl.pkl': 0, 'elasticNet_kl.pkl': 0, 'lasso_kl.pkl': 0, 'svm_linear_kl.pkl': 0,'svm_rbf_kl.pkl': 0}
weight_ls_kl=[weights1_kl,weights2_kl,weights3_kl,weights4_kl,weights5_kl,weights6_kl,weights7_kl,weights8_kl,weights9_kl,weights10_kl,weights11_kl,weight12_kl]

t=0
df = pd.read_excel(score_path_qf)
# 添加新数据的示例
new_row = pd.Series(dtype='object')
df = df.append(new_row, ignore_index=True)
for weights in weight_ls:
    t=t+1
    print(f'--------第{t}种投票结果如下--------')
    # 初始化加权预测结果的存储数组
    weighted_preds_qf = np.zeros(y_qf.shape)
    for model_path in model_paths:
        # print('model_path:',model_path)
        # 获取模型文件的简单名称，如'A.pkl'
        model_name = model_path.split('/')[-1]
        # 加载模型
        model = joblib.load(model_path)
        # 进行预测
        pred_qf = model.predict(X_qf)
        # 根据模型名称获取其权重并计算加权预测结果
        weighted_preds_qf += weights[model_name] * pred_qf
    # print('weighted_preds:\n',weighted_preds)
    # 计算平均误差
    err = abs(y_qf - weighted_preds_qf)
    print('误差列表如下：')
    for i in err.index:
        if err[i]<30:
            print(f'{err[i]:.4},origin={y_qf[i]}')
    mae=np.mean(err)
    print('平均误差：',mae )
    r2 = r2_score(y_qf, weighted_preds_qf)
    print('R^2精度：',r2)
    mape_qf=mean_absolute_percentage_error(y_qf,weighted_preds_qf)
    print('平均绝对百分误差：',mape_qf*100,'%')
    new_data = {'Model': f'{t}', 'R^2': r2, 'MAPE': mape_qf * 100, 'MAE': mae}
    # 将新数据添加到数据框中
    df = df.append(new_data, ignore_index=True)
    # 保存数据框到Excel表格
    df.to_excel(score_path_qf, index=False)
t=0
df = pd.read_excel(score_path_kl)
# 添加新数据的示例
new_row = pd.Series(dtype='object')
df = df.append(new_row, ignore_index=True)
for weights in weight_ls_kl:
    t=t+1
    print(f'--------第{t}种投票结果如下--------')
    # 初始化加权预测结果的存储数组
    weighted_preds_kl = np.zeros(y_kl.shape)
    for model_path in model_paths_kl:
        # print('model_path:',model_path)
        # 获取模型文件的简单名称，如'A.pkl'
        model_name = model_path.split('/')[-1]
        # 加载模型
        model = joblib.load(model_path)
        # 进行预测
        pred_kl = model.predict(X_kl)
        # 根据模型名称获取其权重并计算加权预测结果
        weighted_preds_kl += weights[model_name] * pred_kl
    # print('weighted_preds:\n',weighted_preds)
    # 计算平均误差
    err = abs(y_qf - weighted_preds_kl)
    print('误差列表如下：')
    for i in err.index:
        if err[i]<20:
            print(f'{err[i]:.4},origin={y_qf[i]}')
    mae=np.mean(err)
    print('平均误差：',mae )
    r2 = r2_score(y_kl, weighted_preds_kl)
    print('R^2精度：',r2)
    mape_kl=mean_absolute_percentage_error(y_qf,weighted_preds_kl)
    print('平均绝对百分误差：',mape_kl*100,'%')
    new_data = {'Model': f'{t}', 'R^2': r2, 'MAPE': mape_kl * 100, 'MAE': mae}
    # 将新数据添加到数据框中
    df = df.append(new_data, ignore_index=True)
    # 保存数据框到Excel表格
    df.to_excel(score_path_kl, index=False)