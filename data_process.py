'''
中位数去极值
缺失值处理
行业市值中性化
标准化
PCA降维(可做可不做)
'''

def fillwith_industry_mean(df,date,industry = 'sw_l1'):
    """
    df: 因子数据
    date:日期
    return: 填充空值后的df
    """
    stocklist = df.index.tolist() 
    # 获得股票所属行业
    industry_dict = get_industry(stocklist,date =date)   
    industry_stock = {}
   
    for k,v in industry_dict.items():  # dict.items()函数以列表返回可遍历的(键, 值) 元组数组
        if 'sw_l1' in v.keys():
            industry_stock[k] = v['sw_l1']['industry_name']
        else:
            # 有的股票没有sw_l1行业
            industry_stock[k] = np.nan
    df['industry'] = pd.Series(industry_stock)  # 序列索引为股票
     
    # 行业均值df ,index为行业,列为因子,值为均值
    industry_mean_df = df.loc[df['industry'].notnull()].groupby('industry').mean()
    for factor in df.columns:
        # 获得该因子为空的股票
        null_codes = df.loc[df[factor].isnull(),factor].index.tolist()
        
        for code in null_codes:
            # 如果知道股票所属行业, 不是np.nan
            if isinstance(df.loc[code,'industry'],str): # isinstance()函数来判断一个对象是否是一个已知的类型
                df.loc[code,factor] = industry_mean_df.loc[df.loc[code,'industry'],factor]
    # 无行业用列均值填充
    df.fillna(df.mean(),inplace=True)
    return df.iloc[:,:-1]
    
    
def factor_processing(df,date,industry = 'sw_l1'):
    # 中位数去极值
    df = winsorize_med(df, scale=5,inf2nan=False,axis=0)
    # 行业均值填充缺失值
    df = fillwith_industry_mean(df,date,industry)
    # 行业市值中性化
    df = neutralize(df,['sw_l1', 'market_cap'] ,date=str(date),axis=0)
    # 标准化
    df = standardlize(df,axis=0)
    return df
    
# 已经保存了处理好的数据,可以直接使用,这一步只需运行一次
processed_factor_dict = {}
for date,factor_df in original_factor_dict.items():
    processed_df = factor_processing(factor_df,date)
    processed_df.dropna(inplace=True)
    processed_factor_dict[date] = processed_df
    

# 保存处理后的因子数据
with open('processed_factor_dict.pkl','wb') as pf:
    pickle.dump(original_factor_dict,pf)
with open('processed_factor_dict.pkl','rb') as pf:
    processed_factor_dict = pickle.load(pf)
 
 
 '''
 生成训练集测试集
 '''
 

# 计算月收益率
def get_return_monthly(stocklist,start,end):
    close_start = get_price(stocklist, count = 1, end_date=start,fields='close',panel=False).set_index('code')[['close']]
    close_end = get_price(stocklist, count = 1, end_date=end,fields='close',panel=False).set_index('code')[['close']]
    df = pd.concat([close_start,close_end],axis=1)
    df['return'] = (df.iloc[:,1]-df.iloc[:,0])/df.iloc[:,0]
    return df['return']
 
def label_data(data,dropna=True):
    '''将数据打标签,收益前0.3为1,后0.3为0'''
    # 初始化分类列
    data['return_bin'] = np.nan
    # 将下期收益从大到小排序
    data = data.sort_values(by='return', ascending=False)
    # [总样本*前0.3,总样本*后0.3]
    n_stock_select = np.multiply(percent_select, data.shape[0]) #数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
    # 取整操作
    # 取整后虽然数值为整数，但是在存储格式上仍为浮点数，
    # 因此还需要使用 astype(int)强制转换成整数格式。
    n_stock_select = np.around(n_stock_select).astype(int)
    # 收益前30% 赋值1,后30%赋值0
    data.iloc[:n_stock_select[0], -1] = 1
    data.iloc[-n_stock_select[1]:, -1] = 0
    if dropna:
        data.drop('return',axis=1,inplace=True)
        data = data.dropna()
    return data   
    
    
# 同样这里可以直接使用处理好的训练数据
# 训练集合并 
# 这里取前88个月作为训练集 , 后36个月作为测试集
train_df_all = pd.DataFrame()
for idx,date in enumerate(trade_days_all[:88]):
    df = processed_factor_dict[date].copy()
    df['return'] = get_return_monthly(df.index.tolist(),date,trade_days_all[idx+1])
    df = label_data(df)
    train_df_all = pd.concat([train_df_all,df])
    
# 保存训练集
with open('train.pkl','wb') as pf:
    pickle.dump(train_df_all,pf)
# with open('train.pkl','rb') as pf:
#     train_df_all = pickle.load(pf)


# 测试集
test_dict = {}
for idx,date in enumerate(trade_days_test[:-1]):
    df = processed_factor_dict[date].copy()
    df['return'] = get_return_monthly(df.index.tolist(),date,trade_days_test[idx+1])
    df = label_data(df,dropna=False)
    test_dict[date] = df
    
# 保存测试集
with open('test.pkl','wb') as pf:
    pickle.dump(test_dict,pf)
# with open('test.pkl','rb') as pf:
#     test_dict = pickle.load(pf)
