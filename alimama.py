import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import xgboost as xgb
from  sklearn.preprocessing import LabelEncoder
lbl=LabelEncoder()
import time



train_set=pd.read_csv('input/round1_ijcai_18_train_20180301/round1_ijcai_18_train_20180301.txt',sep=' ')
test_set=pd.read_csv('input/round1_ijcai_18_test_a_20180301/round1_ijcai_18_test_a_20180301.txt',sep=' ')



def output_image(train_set):
    
    plt.grid(True)

    for col_name in train_set.columns:
        if train_set[col_name].dtype!='float64':
            fig = plt.figure(figsize=(15, 15), dpi=100)
            trade_rate = train_set.groupby(col_name)['is_trade'].mean()
            print("----------------------------------------")
            print("column", col_name, "trade rate is:")
            print(trade_rate.describe())
            print("----------------------------------------")
            print("column", col_name, "influce var is:")
            print(trade_rate.var())
            print("#####################################################")

            trade_rate = train_set.groupby(col_name)['is_trade'].mean()

            trade_rate.plot(kind='bar', color='g')
            plt.savefig('output/' + col_name + '.jpg', dpi=100, figsize=(20, 20))
            plt.close()




train_set['source']='train'
test_set['source']='test'
train_df=pd.concat([train_set,test_set],ignore_index=True)
m,n=train_df.shape

#缺失值分析
miss_count=train_df.apply(lambda x:sum(x==-1))
miss_count.plot(kind='bar')
miss_rate=(miss_count/m)

miss_rate.plot(kind='bar')
plt.show()


#特征分析

category_var=['context_page_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level'\
              ,'item_pv_level','user_gender_id','user_age_level','user_occupation_id','user_star_level','shop_review_num_level',
              'shop_star_level']
for v in category_var:
    print (v,'不同取值出现的次数：')
    print (train_df[v].value_counts())

   
category_var=['context_page_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level'\
              ,'item_pv_level','user_gender_id','user_age_level','user_occupation_id','user_star_level','shop_review_num_level',
              'shop_star_level']
for v in category_var:
    print ('*************************************')
    print (v,'成交率：')
    print (train_df.groupby(v)['is_trade'].mean())
    print ('成交率的方差为：')
    print (train_df.groupby(v)['is_trade'].mean().var())



#特征处理

# 缺失值（除了性别全用众数顶上）
category_var=['context_page_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level'\
              ,'item_pv_level','user_gender_id','user_age_level','user_occupation_id','user_star_level','shop_review_num_level',
              'shop_star_level']
for v in category_var:
    top_num=train_df[v].value_counts()[:1].values[0]
    train_df[v][train_df[v]==-1]=top_num


#组合特征
def connact(x,y):
    
    return str(x)+'_'+str(y)

user_shop=[]
for i in range(len(train_df)):
    user_shop.append(connact(train_df.shop_id.values[i],train_df.user_id.values[i]))


train_df['user_shop']=user_shop

train_df.pivot_table()


#把数量少的类目全都归位一类low_level
category_var=['item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level'\
              ,'item_pv_level','user_age_level','user_occupation_id','user_star_level','shop_review_num_level','shop_star_level']
for v in category_var:
        value_counts = train_df[v].value_counts()/m
        be_replace = ((value_counts[value_counts < 0.005])).index.tolist()
        if len(be_replace) > 0:
            train_df[v] = train_df[v].replace(be_replace, be_replace[0])

# 类目编码
category_var=['context_page_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level'\
              ,'item_pv_level','user_gender_id','user_age_level','user_occupation_id','user_star_level','shop_review_num_level','shop_star_level']
for v in category_var:
    train_df[v]=lbl.fit_transform(train_df[v])


# list变量处理
list_var=['item_category_list','item_property_list','predict_category_property']

#TODO 排序处理
new_list_var=[]
#取前0-9位置数据的数作为一个编号属性
for i in range(10):
    new_list_var.append('item_property_list' + str(i))
    train_df['item_property_list' + str(i)] = lbl.fit_transform(train_df['item_property_list'].map(
        lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))


for i in range(5):
    new_list_var.append('predict_category_property' + str(i))
    train_df['predict_category_property' + str(i)] = lbl.fit_transform(train_df['predict_category_property'].map(
        lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))


# 取前1-2位置数据的数作为一个编号属性
for i in range(1, 3):
    new_list_var.append('item_category_list' + str(i))
    train_df['item_category_list' + str(i)] = lbl.fit_transform(train_df['item_category_list'].map(
        lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  # item_category_list的第0列全部都一样
del train_df['item_property_list']
del train_df['predict_category_property']
del train_df['item_category_list']    



# 处理时间戳
print('process time ...')
format = '%Y-%m-%d %H:%M:%S'
date_time = train_df.context_timestamp.apply(lambda x: pd.to_datetime(time.strftime(format, time.localtime(x))))
train_df['day'] = date_time.dt.day
train_df['hour'] = date_time.dt.hour
#是否是工作时间
#train_df['is_work']=train_df['hour'].apply(lambda x:1 if x>18 else 0)
#处理一下
train_df['week'] = date_time.dt.dayofweek
#是否周末
#train_df['is_sunday']=train_df['week'].apply(lambda x: 1 if x>4 else 0)
del train_df['context_timestamp']
print ('time finished')


#删除id 列
train_df[train_df.source=='test'].instance_id.to_csv('output/submit_ids_csv',index=None)
del train_df['context_id']
del train_df['instance_id']
del train_df['item_id']
del train_df['user_id']
del train_df['shop_id']
del train_df['date_time']

#特征输出

train_output=train_df[train_df['source']=='train']
test_output=train_df[train_df['source']=='test']

del train_output['source']
del test_output['source']
output_image(train_output)
del test_output['is_trade']
train_output.to_csv('output/train_set.csv',index=None)
test_output.to_csv('output/test_set.csv',index=None)




