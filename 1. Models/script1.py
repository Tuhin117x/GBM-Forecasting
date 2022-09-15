import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

train_org=pd.read_csv("train_m.csv",sep=',')
test_org=pd.read_csv("holdout_features.csv",sep=',')

filt_df=train_org[['asin','market','total_unit_sold','product_company_type_source','retailer','category','review_rating']]
filt_test_df=test_org[['asin','market','product_company_type_source','retailer','category','review_rating']]

avg_rating=filt_df.groupby(['asin','market','product_company_type_source','retailer','category','total_unit_sold'],as_index=False)[('review_rating')].mean()
avg_rating=pd.DataFrame(avg_rating,columns=['asin','market','product_company_type_source','retailer','category','total_unit_sold','review_rating'])
avg_rating_predict=filt_test_df.groupby(['asin','market','product_company_type_source','retailer','category'],as_index=False)[('review_rating')].mean()

train=avg_rating.copy()
test=avg_rating_predict.copy()

#Map features and Target variables for the LGB model
target=['total_unit_sold']
#features=['asin','market','product_company_type_source','retailer','category','subcategory','review_rating']
#cat_features=['asin','market','product_company_type_source','retailer','category','subcategory']
features=['asin','market','product_company_type_source','retailer','category','review_rating']
cat_features=['asin','market','product_company_type_source','retailer','category']



#Define Categorical values for non-quantitative data
label_encoder_asin_train = LabelEncoder()
label_encoder_market_train = LabelEncoder()
label_encoder_product_company_type_source_train = LabelEncoder()
label_encoder_retailer_train = LabelEncoder()
label_encoder_category_train = LabelEncoder()
#label_encoder_subcategory_train = LabelEncoder()

train['asin'] = label_encoder_asin_train.fit_transform(train['asin'])
train['market'] = label_encoder_market_train.fit_transform(train['market'])
train['product_company_type_source'] = label_encoder_product_company_type_source_train.fit_transform(train['product_company_type_source'])
train['retailer'] = label_encoder_retailer_train.fit_transform(train['retailer'])
train['category'] = label_encoder_category_train.fit_transform(train['category'])
#train['subcategory'] = label_encoder_subcategory_train.fit_transform(train['subcategory'])

label_encoder_test_asin = LabelEncoder()
label_encoder_test_market = LabelEncoder()
label_encoder_test_product_company_type_source = LabelEncoder()
label_encoder_test_retailer = LabelEncoder()
label_encoder_test_category = LabelEncoder()
#label_encoder_test_subcategory = LabelEncoder()

test['asin'] = label_encoder_test_asin.fit_transform(test['asin'])
test['market'] = label_encoder_test_market.fit_transform(test['market'])
test['product_company_type_source'] = label_encoder_test_product_company_type_source.fit_transform(test['product_company_type_source'])
test['retailer'] = label_encoder_test_retailer.fit_transform(test['retailer'])
test['category'] = label_encoder_test_category.fit_transform(test['category'])
#test['subcategory'] = label_encoder_test_subcategory.fit_transform(test['subcategory'])




#Split data into training and validation dataset
dataset1=train.iloc[2:,:]
dataset2=train.iloc[:2,:]
# print(dataset1)
# print(dataset2)
x_train = dataset1[features].fillna(value=0)
y_train = dataset1[target].fillna(value=0)
x_valid=dataset2[features].fillna(value=0)
y_valid=dataset2[target].fillna(value=0)
dtrain=lgb.Dataset(x_train,y_train)
dvalid=lgb.Dataset(x_valid,y_valid)

#Define parameters for the LGB Model
params = {'metric': 'rmse',
          'num_leaves': 255,
          'learning_rate': 0.005,
          'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'bagging_freq': 5,
          'force_col_wise' : True,
          'random_state': 10}
 
#Train the Boosting based ensemble model
lgb_model=lgb.train(params=params,
                    train_set=dtrain,
                    num_boost_round=1500,
                    valid_sets=(dtrain, dvalid),
                    early_stopping_rounds=150,
                    categorical_feature=cat_features,
                    verbose_eval=100)   
 
#Predict the power generated output variable
test1=test[features]
test1=test1.fillna(0)
test1['preds']=lgb_model.predict(test1)

preds_1=test1[['asin','market','preds']]
preds_1.columns=['asin','market','preds']
preds_1['asin']=label_encoder_test_asin.inverse_transform(preds_1['asin'])
preds_1['market']=label_encoder_test_market.inverse_transform(preds_1['market'])
 
#Export the predicted values to a CSV for final submission
preds_1.to_csv('my_submission.csv',index=False)

preds_1.head(10)