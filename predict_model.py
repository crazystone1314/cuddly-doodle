# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import catboost as cb


# 原始数据路径
training_path = 'D://train_data.csv'
source_data_path = 'D://preliminary-testing.csv'

# --------读取原始数据-------
# training数据
train_data = pd.read_csv(training_path, names=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand'])
# preliminary-testing数据
test_data = pd.read_csv(source_data_path, names=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5'])



# --------将J、Q、K映射成11、12、13------
def transform_jkq(x):
    if x == 'J':
        return 11
    elif x == 'Q':
        return 12
    elif x == 'K':
        return 13
    else:
        return x

# train_data数据处理
train_data['C1'] = train_data['C1'].apply(transform_jkq)
train_data['C2'] = train_data['C2'].apply(transform_jkq)
train_data['C3'] = train_data['C3'].apply(transform_jkq)
train_data['C4'] = train_data['C4'].apply(transform_jkq)
train_data['C5'] = train_data['C5'].apply(transform_jkq)

# preliminary-testing数据处理
test_data['C1'] = test_data['C1'].apply(transform_jkq)
test_data['C2'] = test_data['C2'].apply(transform_jkq)
test_data['C3'] = test_data['C3'].apply(transform_jkq)
test_data['C4'] = test_data['C4'].apply(transform_jkq)
test_data['C5'] = test_data['C5'].apply(transform_jkq)


# -------将C、D、H、S 映射为1、2、3、4--------
encode_map = {'C':1, 'D':2, 'H':3,'S':4}
# training数据处理
train_data['S1'] = train_data['S1'].map(encode_map)
train_data['S2'] = train_data['S2'].map(encode_map)
train_data['S3'] = train_data['S3'].map(encode_map)
train_data['S4'] = train_data['S4'].map(encode_map)
train_data['S5'] = train_data['S5'].map(encode_map)

# preliminary-testing数据处理
test_data['S1'] = test_data['S1'].map(encode_map)
test_data['S2'] = test_data['S2'].map(encode_map)
test_data['S3'] = test_data['S3'].map(encode_map)
test_data['S4'] = test_data['S4'].map(encode_map)
test_data['S5'] = test_data['S5'].map(encode_map)


# --------计算四种花色的数量和13种排名的有无---------
def bincount2D_vectorized(a):
    N = a.max()+1
    a_offs = a + np.arange(a.shape[0])[:,None]*N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1,N)

# training数据处理
# 计算四种花色的数量
S_training = train_data.iloc[:, [0, 2, 4, 6, 8]].astype(int)
S_training = pd.DataFrame(bincount2D_vectorized(S_training.values),columns=['suitCount0','suitCount1','suitCount2','suitCount3','suitCount4'])
train_data = pd.merge(train_data, S_training, how='left', left_index=True, right_index=True).drop(['suitCount0'], axis=1)
#计算13种排名的有无
R_training = train_data.iloc[:, np.arange(1, 10, 2)].astype(int)
cols = ['rank{}'.format(x) for x in range(0,14,1)]
R_training = pd.DataFrame(bincount2D_vectorized(R_training.values),columns=cols)
train_data = pd.merge(train_data, R_training, how='left', left_index=True, right_index=True).drop(['rank0'], axis=1)

# preliminary-testing数据处理
#计算13种排名的有无
S_source_data = test_data.iloc[:, [0, 2, 4, 6, 8]].astype(int)
S_source_data = pd.DataFrame(bincount2D_vectorized(S_source_data.values),columns=['suitCount0','suitCount1','suitCount2','suitCount3','suitCount4'])
test_data = pd.merge(test_data, S_source_data, how='left', left_index=True, right_index=True).drop(['suitCount0'], axis=1)
#计算13种排名的有无
R_source_data = test_data.iloc[:, np.arange(1, 10, 2)].astype(int)
cols = ['rank{}'.format(x) for x in range(0,14,1)]
R_source_data = pd.DataFrame(bincount2D_vectorized(R_source_data.values),columns=cols)
test_data = pd.merge(test_data, R_source_data, how='left', left_index=True, right_index=True).drop(['rank0'], axis=1)


# ------各种排名的种类数------

# training数据处理
R_training = train_data.loc[:, ['rank{}'.format(n) for n in range(1, 14, 1)]].astype(int)
R_training = pd.DataFrame(bincount2D_vectorized(R_training.values),columns=['rankCount{}'.format(n) for n in range(0,5,1)])
train_data = pd.merge(train_data, R_training, how='left', left_index=True, right_index=True).drop(['rankCount0'], axis=1)

# preliminary-testing数据处理
R_source_data = test_data.loc[:, ['rank{}'.format(n) for n in range(1, 14, 1)]].astype(int)
R_source_data = pd.DataFrame(bincount2D_vectorized(R_source_data.values),columns=['rankCount{}'.format(n) for n in range(0,5,1)])
test_data = pd.merge(test_data, R_source_data, how='left', left_index=True, right_index=True).drop(['rankCount0'], axis=1)


# ------13种排名各排名之间的差值的绝对值-----

# training数据处理
train_data['diff1_13'] = np.abs(train_data['rank1'] - train_data['rank13'])
for i in range(2,14,1):
    train_data['diff{}_{}'.format(i, i - 1)] = np.abs(train_data['rank{}'.format(i)] - train_data['rank{}'.format(i - 1)])
# train_data['diff13_1'] = np.abs(train_data['rank13'] - train_data['rank1'])

# preliminary-testing数据处理
test_data['diff1_13'] = np.abs(test_data['rank1'] - test_data['rank13'])
for i in range(2,14,1):
    test_data['diff{}_{}'.format(i, i - 1)] = np.abs(test_data['rank{}'.format(i)] - test_data['rank{}'.format(i - 1)])
# train_data['diff13_1'] = np.abs(train_data['rank13'] - train_data['rank1'])


# ------删除原始特征和13种花色的有无-----

# training数据处理
train_data = train_data.drop(['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5'], axis=1)
train_data = train_data.drop(['rank{}'.format(n) for n in range(1, 14, 1)], axis=1)

# preliminary-testing数据处理
test_data = test_data.drop(['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5'], axis=1)
test_data = test_data.drop(['rank{}'.format(n) for n in range(1, 14, 1)], axis=1)


# --------训练模型并用模型预测数据--------
X = train_data.drop(['hand'], axis=1)
y = train_data.hand

params = {
            'l2_leaf_reg':0.8,
            'learning_rate':0.09,
            'depth':11,
            'iterations':250
         }
cat = cb.CatBoostClassifier(loss_function='MultiClassOneVsAll', random_seed=1234)
# 设置模型参数
cat.set_params(**params)
# 训练模型
cat_model = cat.fit(X, y, verbose=False)
#用模型进行预测
preds_class = cat_model.predict(test_data, prediction_type='Class')
result = pd.DataFrame(preds_class)
# 将结果转化为整型
result_1 = result[0].apply(int)
result_2 = pd.DataFrame(result_1)
# 将数据保存到文件dsjyycxds_preliminary.txt中
result_2.to_csv('D://dsjyycxds_preliminary.txt', index=False, header=False)