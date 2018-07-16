#encoding: utf-8

#调用的库
import numpy as np # science cal
import pandas as pd # data ansys
import matplotlib.pyplot as plt

#import matplotlib as mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']
from pandas import Series,DataFrame

#数据读取，训练集
data_train = pd.read_csv("~/Documents/data/train.csv")
data_train

data_train.info()

#plot分析画图
fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0)) # 画成整张图
data_train.Survived.value_counts().plot(kind='bar') # 柱状图
plt.title(u"存活分布 (1为存活)") #标题
plt.ylabel(u"人数") #  纵坐标名称

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u"舱位等级")
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄存活分布")

plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱',u'二等舱',u'三等舱'),loc='best')#set our legend for our gragh

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各口岸上船人数")
plt.ylabel(u"人数")

#生还者性别分布
data_train.groupby(['Sex','Survived'])['Survived'].count()
data_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

plt.show()
