#### 提取特征
X = train_df_all.iloc[:,:-1]
# 提取标签
y = train_df_all['return_bin']
#################################################
#### F检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

F,p_values = f_classif(X,y)
# 选择p值小于0.01的特征
k = F.shape[0] - (p_values>0.01).sum()

# 得到经过F检验处理过后的特征矩阵
selector = SelectKBest(f_classif, k=k).fit(X, y)
X_test = selector.transform(X)

##################################################
#### 互信息法
from sklearn.feature_selection import mutual_info_classif
# 每个特征和标签之间的互信息量估计
# 越接近0表示越无关,越接近1表示越相关
result = mutual_info_classif(X,y)
k = result.shape[0]-sum(result<=0)

# 得到经过互信息处理过后的特征矩阵
selector = SelectKBest(mutual_info_classif, k=k).fit(X, y)
X_mic = selector.transform(X)

####################################################  
### Embedded嵌入法
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
# 这里使用随机森林做特征筛选
RFC_ = RFC(n_estimators =100,max_depth=3,random_state=0)
# 可以看出特征的重要性
result = RFC_.fit(X,y).feature_importances_
print(result)

selector = SelectFromModel(RFC_,threshold=0.005).fit(X,y)
X_embedded = selector.transform(X)
X_embedded.shape

####################################################
# 递归特征消除法(Recursive feature elimination,RFE)
from sklearn.feature_selection import RFE

# n_features_to_select : 想要选择的特征个数
# step : 每次迭代希望移除的特征个数
selector = RFE(RFC_,n_features_to_select=20,step=1).fit(X,y)

# 重要性排名
print(selector.ranking_)
X_wrapper = selector.transform(X)
