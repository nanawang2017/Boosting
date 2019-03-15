
from numpy import loadtxt
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# 载入数据集
dataset=loadtxt('data.csv',delimiter=',')
X=dataset[:,0:8]
Y=dataset[:,8]

# 把数据集拆分成训练集和测试集
seed=7
test_size=0.33
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)

# 拟合XGBoost模型
model=XGBClassifier()
model.fit(X_train,Y_train)

# 对测试集做预测
y_predict=model.predict(X_test)
prediction=[round(value) for value in y_predict ]
# round 返回浮点数x的四舍五入值。
accuracy=accuracy_score(Y_test,prediction)
# 输出准确率
print('Accuracy:%.2f%%'%(accuracy*100))

