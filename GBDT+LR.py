"""
为了降低过拟合的风险，将训练集中的数据划分为两部分，

一部分数据用于训练GBDT模型，另一部分数据通过训练好

的GBDT模型得到新特征以训练LR模型。

算法引申
这一部分将会简单提供一些GBDT+LR混合模型的引申思路，希望对大家实际使用时有所裨益。

用FFM模型替代LR模型：

直接将GBDT所得特征输入FFM模型；

用XGBoost模型替代GBDT模型；

将stacking模型学习层中的GBDT交叉检验；

GBDT和LR模型使用model fusion，而不是stacking


"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


X_gbdt,X_lr,y_gbdt,y_lr=train_test_split(X,y,test_size=0.5)
# 通过sklearn中的GradientBoostingClassifier得到GBDT
gbdt=GradientBoostingClassifier()
# 使用GBDT模型的fit方法训练模型
gbdt.fit(X_gbdt,y_gbdt)
# GBDT模型的apply方法得到新特征。
leaves=gbdt.apply(X_lr)[:,:,0]
# 使用sklearn.preprocessing中的OneHotEncoder将GBDT所得特征进行One-hot
features_trans=OneHotEncoder.fit_transform(leaves)
# LR进行分类
# 用经过离散化处理的新特征训练LR模型并得到预测结果。
lr=LogisticRegression()
lr.fit(features_trans,y_lr)
lr.predict_proba(features_trans)[:,1]