from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score,roc_curve

def group_return(df): 
    """
    分层回测函数
    """
    length = df.shape[0]//5
    df = df.sort_values('predict', ascending=False)
    g1 = df.iloc[:length,:]['return'].mean()
    g2 = df.iloc[length:2*length,:]['return'].mean()
    g3 = df.iloc[2*length:-2 *length,:]['return'].mean()
    g4 = df.iloc[-2*length:-length,:]['return'].mean()
    g5 = df.iloc[-length:,:]['return'].mean()
    return [g1,g2,g3,g4,g5]
    
    
def visualization(result_df):
    '''
    :param result_df: 分组收益结果
    '''
    try:
        # 计算累计收益
        Result_cum_return = (result_df + 1).cumprod()
        Result_cum_return.index = [str(i) for i in Result_cum_return.index]
        # 创建画布
        plt.figure(figsize=(10,6))
        # 定义画图样式
        plt.style.use('ggplot')
        plt.plot(Result_cum_return['group1'], label="group1")
        plt.plot(Result_cum_return['group2'], label="group2")
        plt.plot(Result_cum_return['group3'], label="group3")
        plt.plot(Result_cum_return['group4'], label="group4")
        plt.plot(Result_cum_return['group5'], label="group5")
        plt.plot(Result_cum_return['benchmark'],ls='--',label="benchmark")
        plt.xticks(rotation=90,fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='x')
        plt.legend(fontsize='large')
    except Exception as e :
        print("作图失败!",e)
    else:
        print("作图成功!")


def evaluation(group_df):
    index = ['累计收益率','年化收益率','最大回撤','夏普比率','年化超额收益率','月最大超额收益','跑赢基准月份占比','正收益月份占比','信息比率']
    eval_df = pd.DataFrame(np.zeros((9,6)),index=index,columns=['第一组','第二组','第三组','第四组','第五组','比较基准'])
    ret_bc = group_df.iloc[:,-1]
    for i in range(len(group_df.columns)):
        ret = group_df.iloc[:,i]
        n = len(group_df.index)
        # 累计收益率
        return_cump = np.around((ret+1).cumprod()[-1] * 100, 1)
        eval_df.iloc[0, i] = str(np.around(return_cump-100,1)) + '%'
        # 年化收益率
        annul = (return_cump / 100) ** (12 / n) - 1
        eval_df.iloc[1, i] = str(np.round(annul * 100, 2)) + '%'
        # 最大回撤
        cummax = (group_df + 1).cumprod().iloc[:, i].cummax()
        maxback = ((cummax - (group_df + 1).cumprod().iloc[:, i]) / cummax).max()
        eval_df.iloc[2, i] = str(np.around(maxback*100, 2)) + '%'
        # 夏普比率
        eval_df.iloc[3, i] = np.around((annul - 0.04) / ret.std(), 2)
        # 年化超额收益率
        alpha = (ret - ret_bc + 1).cumprod()[-1]
        alpha_ann = (alpha) ** (12 / n) - 1
        eval_df.iloc[4, i] = str(np.round(alpha_ann * 100, 2)) + '%'
        # 月最大超额收益
        eval_df.iloc[5, i] = str(np.round((ret - ret_bc).max() * 100, 2)) + '%'
        # 跑赢基准概率
        eval_df.iloc[6, i] = str(np.round((ret > ret_bc).sum() / n * 100, 2)) + '%'
        # 正收益月份占比
        eval_df.iloc[7, i] = str(np.round((ret > 0).sum() / n * 100, 2)) + '%'
        # 信息比率
        ann_bc = (ret_bc+1).cumprod()[-1]**(12/n)-1
        std = (ret-ret_bc).std()
        eval_df.iloc[8, i] = np.around((annul - ann_bc)/std,2) if i!=len(group_df.columns)-1 else np.NAN
    return eval_df
 
 
 
def model_evaluation(model,test_dict,trade_days_test,selector):
    """
    模型评价函数
    model: 训练好的模型
    test_dict : 测试集字典
    trade_days_test : 测试集日期列表
    selector: 特征筛选器
    返回: 精确度,auc,每组每期收益
    """
    accuracy_monthly = []
    auc_monthly = []
    group_return_dict = {}
    # 遍历测试集日期
    for idx,date in enumerate(trade_days_test[:-1]):
        # 拿到当期测试集数据
        df = test_dict[date].copy()
        # 预测概率
        if hasattr(model,'decision_function'):
            y_predict = model.decision_function(selector.transform(df.iloc[:,:-2]))
        else:
            y_predict = model.predict_proba(selector.transform(df.iloc[:,:-2]))[:,1]
        x_test = selector.transform(df[df['return_bin'].notnull()].iloc[:,:-2])
        y_test = df[df['return_bin'].notnull()].iloc[:,-1]
        # 当期精确度
        accuracy_monthly.append(model.score(x_test,y_test))
        # 当期auc
        auc_monthly.append(roc_auc_score(y_test,y_predict[df['return_bin'].notnull()]))

        # 根据预测的结果将股票分成5组,计算每组收益
        df['predict'] = y_predict
        grouplist = group_return(df)
        # 计算bench下期收益
        bench = get_price(INDEX,date,trade_days_test[idx+1],fields=['close'],panel=False)['close']
        bench = (bench[-1]-bench[0])/bench[0]
        grouplist.append(bench)
        group_return_dict[date] = grouplist
    group_df = pd.DataFrame(group_return_dict,index=['group1','group2','group3','group4','group5','benchmark']).T
    return accuracy_monthly,auc_monthly,group_df
        
def plot_accuracy_auc(accu_list,auc_list):
    print("精确度均值:",mean(accu_list))
    print("AUC均值:",mean(auc_list))

    plt.style.use('ggplot')
    plt.figure(figsize=(10,6))
    xticks = [str(i) for i in trade_days_test[:-1]]
    plt.plot(xticks,accu_list,label='accuracy')
    plt.plot(xticks,auc_list,label='AUC')
    plt.xticks(rotation=90)
    plt.grid(axis='x')
    plt.legend(fontsize='large')
    plt.show()
    
    
########KNN
'''
 样本内正确率和AUC
'''
 from sklearn.neighbors import KNeighborsClassifier as KNN
# 实例化模型
knn = KNN(n_neighbors=10)
params = {'n_neighbors':list(range(5,11))}
knn_GS = GridSearchCV(estimator=knn,param_grid=params,cv=5)
knn_GS.fit(X,y)

# 输出最佳模型结果
print(knn_GS.best_score_)
# 输出最佳模型参数
print(knn_GS.best_params_)  # 跑出来的最佳结果是10

print('样本内交叉验证 : ')
accuracy_train = cross_val_score(knn,X_wrapper,y,scoring='accuracy',cv=5).mean()  # 评价指标：准确率
print('\t','accuracy = %.4f' % (accuracy_train)) #accuracy = 0.5143

auc_train = cross_val_score(knn,X_wrapper,y,scoring='roc_auc',cv=5).mean()   # auc面积
print('\t','auc = %.4f' % (auc_train))    # auc = 0.5199

knn.fit(X_wrapper,y)
with open('knn.pkl','wb') as pf:  # "wb" 以二进制写方式打开，只能写文件， 如果文件不存在，创建该文件；如果文件已存在，先清空，再打开文件
    pickle.dump(knn,pf) #将对象KNN保存到文件pf中去，file必须有write()接口


######样本外选股策略表现
accu_1,auc_1,group_df1 = model_evaluation(knn,test_dict,trade_days_test,selector)  # 模型评估
# 样本外accuracy和auc
plot_accuracy_auc(accu_1,auc_1)
# 净值曲线
visualization(group_df1)
# 策略各项指标
eval1 = evaluation(group_df1)
print(eval1)

######逻辑回归
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
params = {'C':[0.001,0.01,0.1,1,10,100,1000]
            ,'max_iter':[100,200,300]
            ,'penalty' : ['l1','l2'] }
lg_GS = GridSearchCV(estimator=lg,param_grid=params,cv=5)
lg_GS.fit(X_wrapper,y)

# 输出最佳模型结果
print(lg_GS.best_score_)   #0.547038188631109
# 输出最佳模型参数
print(lg_GS.best_params_)  #{'C': 1000, 'max_iter': 300, 'penalty': 'l1'}
lg = LogisticRegression(C=1000,max_iter=300,penalty='l1')
print('样本内交叉验证 : ')
accuracy_train = cross_val_score(lg,X_wrapper,y,scoring='accuracy',cv=5).mean()
print('\t','accuracy = %.4f' % (accuracy_train))  #accuracy = 0.5458

auc_train = cross_val_score(lg,X_wrapper,y,scoring='roc_auc',cv=5).mean()
print('\t','auc = %.4f' % (auc_train))   #auc = 0.5637

lg.fit(X_wrapper,y)
with open('lg.pkl','wb') as pf:
    pickle.dump(lg,pf)

# 样本外选股策略表现
accu_2,auc_2,group_df2 = model_evaluation(lg,test_dict,trade_days_test,selector)
# 样本外accuracy和auc
plot_accuracy_auc(accu_2,auc_2)

# 净值曲线
visualization(group_df3)

eval3 = evaluation(group_df3)
print(eval3)

#############高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
bys = GaussianNB()
print('样本内交叉验证 : ')
accuracy_train = cross_val_score(bys,X_wrapper,y,scoring='accuracy',cv=5).mean()
print('\t','accuracy = %.4f' % (accuracy_train))
auc_train = cross_val_score(bys,X_wrapper,y,scoring='roc_auc',cv=5).mean()
print('\t','auc = %.4f' % (auc_train))   

bys.fit(X_wrapper,y)
with open('bys.pkl','wb') as pf:
    pickle.dump(bys,pf)
    
# 样本外选股策略表现
accu_4,auc_4,group_df4 = model_evaluation(bys,test_dict,trade_days_test,selector)
# 样本外accuracy和auc
plot_accuracy_auc(accu_4,auc_4)

# 净值曲线
visualization(group_df4)
eval4 = evaluation(group_df4)
print(eval4)

############5.5 随机森林
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
params = {'n_estimators':[50,100,200,300]
            ,'max_depth':[*range(3,6)]
            ,'min_samples_split' : [2,10,20,30] 
            ,'min_samples_leaf':[1,5,10,20]
         }
rfc_GS = GridSearchCV(estimator=rfc,param_grid=params,cv=5)
rfc_GS.fit(X_wrapper,y)

rfc = RandomForestClassifier(n_estimators=200
                            ,max_depth=5
                            ,min_samples_leaf=20
                            ,min_samples_split=20
                           )
                           
print('样本内交叉验证 : ')
accuracy_train = cross_val_score(rfc,X_wrapper,y,scoring='accuracy',cv=5).mean()
print('\t','accuracy = %.4f' % (accuracy_train))  #accuracy = 0.5474

auc_train = cross_val_score(rfc,X_wrapper,y,scoring='roc_auc',cv=5).mean()
print('\t','auc = %.4f' % (auc_train))  #auc = 0.5667

rfc.fit(X_wrapper,y)
with open('rfc.pkl','wb') as fp:
    pickle.dump(rfc,fp)
    
# 样本外选股策略表现
accu_5,auc_5,group_df5 = model_evaluation(rfc,test_dict,trade_days_test,selector)
# 样本外accuracy和auc
plot_accuracy_auc(accu_5,auc_5)
visualization(group_df5)

eval5 = evaluation(group_df5)
print(eval5)

##############5.6 Adaboost
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier()
params = {
    'base_estimator':[DecisionTreeClassifier(max_depth=5)],
    'n_estimators' : [50,100,200,300],
    'learning_rate': [0.01,0.1,1,10,100]
}
adb_GS = GridSearchCV(estimator=adb,param_grid=params,cv=5)
adb_GS.fit(X_wrapper,y)
# 输出最佳模型结果
print(adb_GS.best_score_) 
# 输出最佳模型参数
print(adb_GS.best_params_)  #'learning_rate': 0.01, 'n_estimators': 200}

# 输出最佳模型结果
print(rfc_GS.best_score_)  # 0.5486327034999602
# 输出最佳模型参数
print(rfc_GS.best_params_) #{'max_depth': 5, 'min_samples_leaf': 20, 'min_samples_split': 20, 'n_estimators': 200}

adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)
                         ,learning_rate=0.01
                        , n_estimators=200)
                        
print('样本内交叉验证 : ')
accuracy_train = cross_val_score(adb,X_wrapper,y,scoring='accuracy',cv=5).mean()
print('\t','accuracy = %.4f' % (accuracy_train))  #accuracy = 0.5452

auc_train = cross_val_score(adb,X_wrapper,y,scoring='roc_auc',cv=5).mean()
print('\t','auc = %.4f' % (auc_train))  #auc = 0.5617

adb.fit(X_wrapper,y)
with open('adb.pkl','wb') as fp:
    pickle.dump(adb,fp)
# 样本外选股策略表现
accu_6,auc_6,group_df6 = model_evaluation(adb,test_dict,trade_days_test,selector)
# 样本外accuracy和auc
plot_accuracy_auc(accu_6,auc_6)
visualization(group_df6)
eval6 = evaluation(group_df6)
print(eval6)

###########5.7 支持向量机
from sklearn.svm import SVC
svm = SVC(C=0.003,kernel='rbf',gamma=0.01)
print('样本内交叉验证 : ')
accuracy_train = cross_val_score(svm,X_wrapper,y,scoring='accuracy',cv=5).mean()
print('\t','accuracy = %.4f' % (accuracy_train))  

auc_train = cross_val_score(svm,X_wrapper,y,scoring='roc_auc',cv=5).mean()
print('\t','auc = %.4f' % (auc_train))

svm.fit(X_wrapper,y)
with open('svm.pkl','wb') as fp:
    pickle.dump(svm,fp)
##样本外选股策略表现  
accu_7,auc_7,group_df7 = model_evaluation(svm,test_dict,trade_days_test,selector)
# 样本外accuracy和auc
plot_accuracy_auc(accu_7,auc_7)
visualization(group_df7)
eval7 = evaluation(group_df7)
print(eval7)

#################5.8 XGBoost
# 样本内正确率和AUC
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=200,max_depth=5)
print('样本内交叉验证 : ')
accuracy_train = cross_val_score(xgb,X_wrapper,y,scoring='accuracy',cv=5).mean()
print('\t','accuracy = %.4f' % (accuracy_train))  

auc_train = cross_val_score(xgb,X_wrapper,y,scoring='roc_auc',cv=5).mean()
print('\t','auc = %.4f' % (auc_train))

xgb.fit(X_wrapper,y)
with open('xgb.pkl','wb') as fp:
    pickle.dump(xgb,fp)
    
###样本外选股策略表现
accu_8,auc_8,group_df8 = model_evaluation(xgb,test_dict,trade_days_test,selector)
# 样本外accuracy和auc
plot_accuracy_auc(accu_8,auc_8)

visualization(group_df8)
eval8 = evaluation(group_df8)
print(eva18)

################总结
models = ['KNN','Logistic','DecisionTree','Bayes','RF','AdaBoost','SVM','XGBoost']
df = pd.DataFrame([[mean(accu_1),mean(accu_2),mean(accu_3),mean(accu_4),mean(accu_5),mean(accu_6),mean(accu_7),mean(accu_8)]
                      ,[mean(auc_1),mean(auc_2),mean(auc_3),mean(auc_4),mean(auc_5),mean(auc_6),mean(auc_7),mean(auc_8)]]
                 ,index=['accuracy','auc']).T
df.index = models

df.plot(kind='bar',figsize=(10,6),ylim=[0.3,0.6],title='Accuracy and AUC of models',fontsize=12)

plt.show()

##################各个模型组合1的净值曲线
group1_all = pd.DataFrame(index=group_df1.index,columns=models+['benchmark'])

for i,df in enumerate([group_df1,group_df2,group_df3,group_df4,group_df5,group_df6,group_df7,group_df8]):
    if i == 7:
        group1_all.iloc[:,i+1] = df['benchmark']
    group1_all.iloc[:,i] = df['group1']

print((group1_all+1).cumprod().iloc[-1,:].sort_values())

plt.figure(figsize=(20,8))
group1_all.index = [str(i) for i in group1_all.index]
plt.plot((group1_all+1).cumprod())
plt.xlabel('times',fontsize=15)
plt.ylabel('net_value',fontsize=15)
plt.xticks(rotation=90,fontsize=12)
plt.yticks([0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1],fontsize=12)
plt.legend(group1_all.columns,fontsize='x-large')
plt.grid(axis='x')
plt.show()


####各个模型组合1的策略评价指标
eval_df = pd.DataFrame(index=eval1.index,columns=models)
for i,df in enumerate([eval1,eval2,eval3,eval4,eval5,eval6,eval7,eval8]):
    if i == 7:
        eval_df['benchmark'] = df['比较基准']
    eval_df.iloc[:,i] = df['第一组']
    
print(eval_df)
