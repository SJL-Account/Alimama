import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
import collections
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import warnings
from  sklearn.ensemble import GradientBoostingClassifier
from   sklearn.cross_validation import train_test_split
import os

mingw_path = 'C:\mingw64\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

warnings.filterwarnings('ignore')
train_set = pd.read_csv('input/round1_ijcai_18_train_20180301/round1_ijcai_18_train_20180301.txt', sep=' ')
test_set = pd.read_csv('input/round1_ijcai_18_test_a_20180301/round1_ijcai_18_test_a_20180301.txt', sep=' ')

train_len = len(train_set)

# 最后提交的id
result_instance_id = test_set['instance_id']

y = train_set.pop('is_trade')
train_df = train_set.append(test_set)

id_df_columns = ['instance_id', 'item_id', 'user_id', 'context_id', 'shop_id']
# id类型删除
train_df.drop(id_df_columns, axis=1, inplace=True)

list_df_columns = ['predict_category_property', 'item_property_list', 'item_category_list']

double_df_columns = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']

lbl = preprocessing.LabelEncoder()


# 处理浮点型特征
def process_float():
    for col_name in double_df_columns:
        print('process ', col_name, '...')
        column = train_df[col_name]
        _mean = column.describe()[1]
        _min = column.describe()[3]
        _25 = column.describe()[4]
        _50 = column.describe()[5]
        _75 = column.describe()[6]
        _max = column.describe()[7]

        column[column == -1] = _mean

        # 等级0
        column[(column >= _min) & (column < _25)] = 10
        # 等级1
        column[(column >= _25) & (column < _50)] = 20
        # 等级2
        column[(column >= _50) & (column < _75)] = 30
        # 等级3
        column[(column >= _75) & (column <= _max)] = 40


def map_hour(x):
    if (x >= 7) & (x <= 12):
        return 1
    elif (x >= 13) & (x <= 20):
        return 2
    else:
        return 3


def process_time():
    # 处理时间戳
    print('process time ...')
    format = '%Y-%m-%d %H:%M:%S'
    date_time = train_df.context_timestamp.apply(lambda x: pd.to_datetime(time.strftime(format, time.localtime(x))))
    train_df['day'] = date_time.dt.day
    train_df['hour'] = date_time.dt.hour.apply(map_hour)
    #是否是工作时间
    #train_df['is_work']=train_df['hour'].apply(lambda x:1 if x>18 else 0)
    train_df['week'] = date_time.dt.week
    #是否周末
    #train_df['is_sunday']=train_df['week'].apply(lambda x: 1 if x>4 else 0)
    del train_df['context_timestamp']


def process_str(x):
    all_str = []
    for i in x:
        all_str.append(i.split(':'))
    all_str = np.array(all_str)
    all_str = all_str.reshape(1, -1)
    return all_str[0]


def count_monmen(x, y):
    return len(np.intersect1d(x, y))


def process_list():
    # 求出个属性类别的长度作为一个属性字段
    train_df['len_item_category'] = train_df['item_category_list'].map(lambda x: len(str(x).split(';')))
    train_df['len_item_property'] = train_df['item_property_list'].map(lambda x: len(str(x).split(';')))


    #取前0-9位置数据的数作为一个编号属性
    for i in range(10):
        train_df['item_property_list' + str(i)] = lbl.fit_transform(train_df['item_property_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))


    for i in range(5):
        train_df['predict_category_property' + str(i)] = lbl.fit_transform(train_df['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))


    # 取前1-2位置数据的数作为一个编号属性
    for i in range(1, 3):
        train_df['item_category_list' + str(i)] = lbl.fit_transform(train_df['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  # item_category_list的第0列全部都一样

    a = train_df.predict_category_property.apply(lambda x: x.split(';')).apply(process_str)

    del train_df['predict_category_property']
    # category
    b = train_df.item_category_list.apply(lambda x: x.split(';'))

    count1 = []
    for i in range(len(train_df)):
        count1.append(count_monmen(a.values[i], b.values[i]))

    train_df['commnon_category'] = count1

    del train_df['item_category_list']

    # property
    c = train_df.item_property_list.apply(lambda x: x.split(';'))
    count2 = []
    for i in range(len(train_df)):
        count2.append(count_monmen(a.values[i], c.values[i]))

    train_df['commnon_property'] = count2

    del train_df['item_property_list']


def process_category():


    for col_name in train_df.columns:
        if train_df[col_name].dtype !='float64':
            # 每种类别的数量
            value_counts = train_df[col_name].value_counts()

            # 数量大于0.001的用概率分布恢复

            print('process ', col_name, '...')

            be_replace = ((value_counts[value_counts < 0.001])).index.tolist()
            # 众数
            nums_top = value_counts.index[0]

            #处理性别缺失值
            if col_name=='user_gender_id':
                train_df['gender_loss']=train_df.apply(lambda x:1 if x==-1 else 0)
                
            #处理-1，用众数代替
            if len(value_counts[value_counts.index == -1]) > 0:
                train_df[col_name] = train_df[col_name].replace(-1, nums_top)

            # 处理无关紧要的值
            if len(be_replace) > 0:
                train_df[col_name] = train_df[col_name].replace(be_replace, be_replace[0])

            #对类别进行编码
            train_df[col_name]=lbl.fit_transform(train_df[col_name].apply(lambda x:str(x)))

def print_information():
    for col_name in train_df.columns:
        value_counts = train_df[:train_len][col_name].value_counts()
        print("column", col_name, "stastic number is:")
        print((value_counts).describe())
        print("----------------------------------------")
        print("column", col_name, "stastic rate number is:")
        print((value_counts / train_len).describe())


def output_image():
    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.grid(True)

    for col_name in train_df.columns:
        trade_rate = train_df[:train_len].groupby(col_name)['is_trade'].mean()
        print("----------------------------------------")
        print("column", col_name, "trade rate is:")
        print(trade_rate.describe())
        print("----------------------------------------")
        print("column", col_name, "influce var is:")
        print(trade_rate.var())
        print("#####################################################")

        trade_rate = train_df[:train_len].groupby(col_name)['is_trade'].mean()

        trade_rate.plot(kind='bar', color='g')

        plt.savefig('output/' + col_name + '.jpg', dpi=100, figsize=(20, 20))
        plt.close()


def print_metrics(true_values, predict_values):
    print("Accuracy:", metrics.accuracy_score(true_values, predict_values))
    print("AUC", metrics.roc_auc_score(true_values, predict_values))
    print("Log Loss:", metrics.log_loss(true_values, predict_values))
    print("Confusion Matrix:", metrics.confusion_matrix(true_values, predict_values))
    print(metrics.classification_report(true_values, predict_values))




# -----------------------------------------特征处理-------------------------------------------------
#process_float()
process_time()
process_list()
process_category()
# print_information()
#train_df = train_df.astype('str')
# train_df=pd.get_dummies(train_df)

train_output_df = train_df[:train_len]
test_output_df = train_df[train_len:]

x = train_output_df.values
test_x = test_output_df.values

# -----------------------------------------调参-------------------------------------------------


print('feature select...')
rdt = RandomForestClassifier(n_estimators=100)
feature_importance = pd.DataFrame()

rdt.fit(x, y)
feature_importance['col'] = train_output_df.columns
feature_importance['score'] = rdt.feature_importances_
sort_ = feature_importance.sort_values(by='score')
col = sort_[sort_.score > 0].col.values.tolist()
sort_.to_csv('feature_importance.csv')
train_output_df= train_output_df[col]
x = train_output_df.values

print ('feature select finished')
# -----------------------------------------建模-------------------------------------------------
print('model fit ...')

'''
#rdt.fit(x, y)
#predict_values = rdt.predict(x)

gbdt=GradientBoostingClassifier()
#gbdt.fit(x,y)
#predict_values = gbdt.predict(x)

lr=LogisticRegression()

rdt = RandomForestClassifier(n_estimators=100 )
clf=rdt

x_train,x_validation,y_train,y_validation=train_test_split(x,y,train_size=.25,random_state=1)
clf.fit(x_train,y_train)
predict_values=clf.predict(x_validation)
print_metrics(y_validation.values, predict_values)

'''
#xgboost
x_train,x_validation,y_train,y_validation=train_test_split(x,y,train_size=.25,random_state=1)

dtrain = xgb.DMatrix(x_train, label=y_train,feature_names=train_output_df.columns.values)
dtest = xgb.DMatrix(x_validation,label=y_validation,feature_names=train_output_df.columns.values)
param = {'bst:max_depth':2, 'bst:eta':0.1, 'silent':0, 'bst:tree_method':'hist','lambda ':100,'bst:predictor':'gpu_predictor',
'objective':'binary:logistic' }
param['nthread'] = 8
param['eval_metric'] = 'logloss'
evallist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 25
early_stopping_rounds=100
bst = xgb.train( param, dtrain, num_round, evallist )
bst.dump_model('dump.raw.txt')

print('model fit finished')
# ----------------------------------------------交叉验证评估-------------------------------------------------
print('cross validation....')
'''
    for max_depth in [10,20,30,40]:
    for n_estimator in [50,100,150,200]:
        rdt = RandomForestClassifier(n_estimators=100 )
        clf=rdt
        x_train,x_validation,y_train,y_validation=train_test_split(x,y,train_size=.25,random_state=1)
        clf.fit(x_train,y_train)
        predict_values=clf.predict(x_validation)
        print_metrics(y_validation.values, predict_values)
'''
print ('cv finished')
# ---------------------------------------------输出-------------------------------------------------------------
x_test=xgb.DMatrix(test_x,feature_names=train_output_df.columns.values)
prob=bst.predict(x_test)
print ('result output....')
#  用方差做权重
result = pd.DataFrame()
result['instance_id'] = result_instance_id
# 这里需要改个名字
result['predicted_score'] =prob
#result[(result > 0.05) & (result < 0.1)] = 0.1
result.to_csv('output/'+time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())+'.txt', sep=' ', float_format='%.1f',index=None)


'''
[0]	eval-logloss:0.457134	train-logloss:0.457191
[1]	eval-logloss:0.327676	train-logloss:0.327775
[2]	eval-logloss:0.247607	train-logloss:0.247743
[3]	eval-logloss:0.195503	train-logloss:0.195672
[4]	eval-logloss:0.16079	train-logloss:0.160989
[5]	eval-logloss:0.137428	train-logloss:0.137643
[6]	eval-logloss:0.121696	train-logloss:0.121922
[7]	eval-logloss:0.111155	train-logloss:0.111408
[8]	eval-logloss:0.104141	train-logloss:0.104387
[9]	eval-logloss:0.099533	train-logloss:0.099774
[10]	eval-logloss:0.096519	train-logloss:0.096687
[11]	eval-logloss:0.094586	train-logloss:0.09473
[12]	eval-logloss:0.093283	train-logloss:0.09335
[13]	eval-logloss:0.092475	train-logloss:0.092484
[14]	eval-logloss:0.091921	train-logloss:0.091903
[15]	eval-logloss:0.091576	train-logloss:0.091532
[16]	eval-logloss:0.091307	train-logloss:0.091233
[17]	eval-logloss:0.091118	train-logloss:0.090995
[18]	eval-logloss:0.090995	train-logloss:0.090831
[19]	eval-logloss:0.090886	train-logloss:0.090693
'''