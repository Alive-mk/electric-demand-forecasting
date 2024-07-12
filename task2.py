#导入模块
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
import tqdm
import sys
import os
import gc
import argparse
import warnings
warnings.filterwarnings('ignore')

#准备数据
train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')

# 合并训练数据和测试数据，并进行排序
data = pd.concat([test, train], axis=0, ignore_index=True)
data = data.sort_values(['id','dt'], ascending=False).reset_index(drop=True)

# 历史平移
for i in range(10,30):
    data[f'last{i}_target'] = data.groupby(['id'])['target'].shift(i)
    
# 窗口统计
data[f'win3_mean_target'] = (data['last10_target'] + data['last11_target'] + data['last12_target']) / 3

# 进行数据切分
train = data[data.target.notnull()].reset_index(drop=True)
test = data[data.target.isnull()].reset_index(drop=True)

# 确定输入特征
train_cols = [f for f in data.columns if f not in ['id','target']]

def time_model(clf, train_df, test_df, cols):
    # 训练集和验证集切分
    trn_x, trn_y = train_df[train_df.dt>=31][cols], train_df[train_df.dt>=31]['target']
    val_x, val_y = train_df[train_df.dt<=30][cols], train_df[train_df.dt<=30]['target']
    # 构建模型输入数据
    train_matrix = clf.Dataset(trn_x, label=trn_y)
    valid_matrix = clf.Dataset(val_x, label=val_y)
    # lightgbm参数
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'min_child_weight': 5,
        'num_leaves': 2 ** 5,
        'lambda_l2': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'learning_rate': 0.05,
        'seed': 2024,
        'nthread' : 16,
        'verbose' : -1,
    }
    # 训练模型
    model = clf.train(lgb_params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], 
                      categorical_feature=[])
    # 验证集和测试集结果预测
    val_pred = model.predict(val_x, num_iteration=model.best_iteration)
    test_pred = model.predict(test_df[cols], num_iteration=model.best_iteration)
    # 离线分数评估
    score = mean_squared_error(val_pred, val_y)
    print(score)
       
    return val_pred, test_pred
    
lgb_oof, lgb_test = time_model(lgb, train, test, train_cols)

# 保存结果文件到本地
test['target'] = lgb_test
test[['id','dt','target']].to_csv('submit.csv', index=None)