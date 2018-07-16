>最近研究了一下kaggle，做了Titanic的项目，用此博客记录一下

# Kaggle-Titanic
>[kaggle链接](https://www.kaggle.com/c/titanic)

>环境：Anaconda，python2.7

>[github源码链接](https://github.com/lex1burner/kaggle-titanic/blob/master/Titanic/base_line.py)

>base_line.py为最终文件

通过观察数据集，这是一个二分类的问题，所以采用逻辑回归模型即可（最初想法）
## 数据可视化
首先拿到数据集需要对其进行可视化查找出可用特征。

## 特征工程（很重要！！）
### 1、预处理
数据集中很多数据是不完整的，我们需要用各种方法来补全数据集。
#### ①众数方法
对于缺失项不多的数据可以用众数填充：
比如 `Embarked`:
```python
#1)Embarked
combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
```
#### ②随机森林预测
缺失项很多，但我们可以用其他特征来预测这个特征的数据作为填充：
比如Age：
```python
##6)Age
###随机森林预测
missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Fare', 'Fare_bin_id', 'Pclass']])

missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
#missing_age_test.info()
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

    #gbm
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(
    gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])

    # model 2 rf
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                               scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
    print('RF Train Error for "Age" Feature Regressor' + str(
    rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])

    # two models merge
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
    # missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)

    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])

    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

    return missing_age_test

combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)
```
#### ③平均数
对于只有一项或者两项缺失的极少数缺失的数据，用平均数来填充：
```python
combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
```
### 2、数值类型转换
由于sklearn中要求都是数字型，所以对非数字型要进行转化：
#### ①dummy
类别变量，比如embarked，只包含S，C，Q三种变量：
```python
emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'], prefix=combined_train_test[['Embarked']].columns[0])
```
#### ②factorize
dummy不能很好的处理Cabin这样具有变量较多的属性，factorize()可以创建一些数字，来表示变量，对应每个类别映射一个ID，这种映射最后只生成一个特征，不会像dummy生成多个特征：
`待完善`
#### ③scaling
是一种映射，将较大的数值映射到较小的范围，比如（-1,1）。
Age的范围比其他属性的范围大很多，这使得Age会有更大的权重，我们需要将其scaling（特征缩放）
```python
from sklearn import preprocessing
assert np.size(df['Age']) == 891
scaler = preprocessing.StandardScaler()
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))
```
#### ④Binning
Fare属性的处理也可以利用上面的方法scaling，也可以用binning。这是一种将连续数据离散化的方法，即将值划分入已经设置好的范围（桶）中。

```python
combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
```
当然数据bin化之后，必须factorize或者dummy。
```python
combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]

fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin_id']).rename(columns = lambda x: 'Fare_' + str(x))
combined_train_test = pd.concat([combined_train_test,fare_bin_dummies_df], axis=1)
combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)

```
### 3、抛弃无用特征
丢到前面一些已经处理过的标签属性，或者中途产生的标签，以及对模型无用的标签。
相关性分析和交叉验证之后，再添加有用标签。
```python
#弃掉无用特征
combined_data_backup = combined_train_test
combined_train_test.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category',
                       'Parch', 'SibSp', 'Ticket', 'Family_Size_Category'],axis=1,inplace=True)
```

## 建立模型
建立简单逻辑回归模型：
```python
from sklearn import linear_model

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(titanic_train_data_X.values, titanic_train_data_Y.values)

#print clf

predictions = clf.predict(titanic_test_data_X)
result = pd.DataFrame({'PassengerId':test_df_org['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("~/Documents/data/base_line_predictions.csv", index=False)
```
## 交叉验证
利用原数据集进行交叉验证
`（待完善）`
## 相关性分析
`坑待填`
## 模型融合
bagging方法进行模型融合：
```python
from sklearn.ensemble import BaggingRegressor

# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                               bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(titanic_train_data_X.values, titanic_train_data_Y.values)

predictions = bagging_clf.predict(titanic_test_data_X)
result = pd.DataFrame({'PassengerId': test_df_org['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
result.to_csv("~/Documents/data/base_bagging_predictions.csv", index=False)
```

## 目录

[ TOC ]

