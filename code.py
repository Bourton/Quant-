### 导入模块

import pandas as pd
import numpy as np
from time import time
import datetime

# 设定最大显示行数、列数为200，宽度
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

from jqdata import *    # 聚宽数据库
from jqfactor import *  # 聚宽因子库
from jqlib.technical_analysis import *  # 聚宽专业分析库
from scipy import stats       # 导入统计模块
import statsmodels.api as sm  # 统计模型
from statsmodels import regression  # 导出回归模型
import pickle  # 数据存储

import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
plt.style.use('ggplot')


########1. 数据获取
#1.1参数设置

# 样本区间
START_DATE = '2010-01-01'
END_DATE = '2020-04-30'

# 指数成份股 ,比较基准 , 这里用ZZ500
INDEX = '399905.XSHE'
# 排名前0.3为1,后0.3为0
percent_select = [0.3, 0.3]  

#1.1.1财务数据
q = query(valuation.code,   # 股票代码
      valuation.market_cap, # 市值
      valuation.circulating_market_cap, # 流通市值
      valuation.pe_ratio,  #市盈率（TTM）
      valuation.pb_ratio,  #市净率（TTM）
      valuation.pcf_ratio, #市现率
      valuation.ps_ratio,  #市销率
      balance.total_assets, #总资产
      balance.total_liability, #总负债
      balance.fixed_assets,  #固定资产
      balance.total_non_current_liability, #长期负债
      income.operating_profit,#营业利润
      income.total_profit, #利润总额
      indicator.net_profit_to_total_revenue, #销售净利率：净利润/营业总收入
      indicator.inc_revenue_year_on_year,  #营业收入增长率（同比）
      indicator.inc_net_profit_year_on_year,#净利润增长率（同比）
      indicator.roe, #净资产收益率
      indicator.roa, #总资产净利率
      indicator.gross_profit_margin #销售毛利率GPM
    )  # 查询字段

#1.1.2聚宽因子列表
# 作为get_factor_values()中的参数factors
jqfactor_list = ['current_ratio', #流动比率
                  'net_profit_to_total_operate_revenue_ttm', #销售净利率
                  'gross_income_ratio', #销售毛利率
                  'roe_ttm', #净资产收益率
                  'roa_ttm', #总资产净利率
                  'total_asset_turnover_rate', # 总资产周转率
                  'net_operating_cash_flow_coverage', #净利润现金含量
                  'net_operate_cash_flow_ttm', #营业外收支净额TTM
                  'net_profit_ttm',  #净利润TTM
                  'cash_to_current_liability',  #现金比率
                  'operating_revenue_growth_rate',# 营业收入增长率
                  'non_recurring_gain_loss', #非经常性损益
                  'operating_revenue_ttm', #营业收入TTM
                  'net_profit_growth_rate', # 净利润增长率
                  'AR', #人气指标
                  'ARBR', #因子 AR 与因子 BR 的差
                  'ATR14', #真实振幅的14日移动平均
                  'VOL5',  #5日平均换手率
                  'VOL60',  #60日平均换手率
                  'Skewness20', # 个股收益的20日偏度
                  'Skewness60'] #个股收益的60日偏度

########1.2获取交易日历(月频)

def get_trade_days_monthly(start,end):
    '''
    获取每月月底的交易日的日历
    注意: 结束日期必须是一个月的最后一天
    :param start:
    :param end:
    :return:  list,每月最后一个交易日的列表,里面元素是datetime.date类型
    '''
    # 日频率的交易日历: Index
    index = get_trade_days(start_date=start, end_date=end)
    # 转成一个df , 索引和值都是交易日(天)
    df = pd.DataFrame(index, index=index)
    # 将Index转成DatetimeIndex
    df.index = pd.to_datetime(df.index)
    # 按月重采样,缺失值用上一个填充,那么刚好值就是想要的每月最后一个交易日
    return list(df.resample('m', how='last').iloc[:,0])

########1.3筛选股票池

# 去除上市距截面期不足n天的股票
def remove_new(stocks,beginDate,n):
    stocklist=[]
    if isinstance(beginDate,str):
        # str转datetime
        beginDate = datetime.datetime.strptime(beginDate, "%Y-%m-%d")
    for stock in stocks:
        start_date=get_security_info(stock).start_date
        # 去除上市距截面期不足n天的股票
        if start_date<(beginDate-datetime.timedelta(days=n)):
            stocklist.append(stock)
    return stocklist
    
# 剔除ST股
def remove_st(stocks,beginDate):
    is_st = get_extras('is_st',stocks,end_date=beginDate,count=1)
    return [stock for stock in stocks if not is_st[stock][0]]

# 剔除每个截面期交易日停牌的股票
def remove_paused(stocks,beginDate):
    is_paused = get_price(stocks,end_date=beginDate, count=1,fields='paused',panel=False)
    return list(is_paused[is_paused['paused']!=1]['code'])
    
def get_stocks_filtered(beginDate,n,indexID=INDEX):
    '''
    获取某一天筛选后的指数成份股
    :param tradedate: 指定某一天
    :param indexID:默认'399905.XSHE'
    :param n : 剔除上市不到n天
    :return:
    '''
    # 获取当天指数成份股列表
    stocklist = get_index_stocks(indexID, date=beginDate)
    stocklist = remove_new(stocklist,beginDate,n)
    stocklist = remove_st(stocklist,beginDate)
    stocklist = remove_paused(stocklist,beginDate)
    return stocklist

########1.4因子数据获取

def get_df_jqfactor(stocklist,factor_list,date):
    '''
    获取聚宽因子
    stocklist：list,股票列表
    factor_list:list,因子列表
    date: 日期， 字符串或 datetime 对象
    output:
    dataframe, index为股票代码，columns为因子
    '''
    # 返回的是一个字典 {'因子1':df1,...}
    factor_data = get_factor_values(securities=stocklist,
                                    factors=factor_list,
                                    count=1,
                                    end_date=date)
    df_jqfactor=pd.DataFrame(index=stocklist)
    
    for factor in factor_data.keys():
        df_jqfactor[factor]=factor_data[factor].iloc[0,:]
    
    return df_jqfactor

def get_newfactors_df(stocklist,df,date):
    """
    stocklist: 股票列表
    df : 原始因子df
    date : 日期
    return : 新的因子df
    
    """
    df_new = pd.DataFrame(index=stocklist)  # 创建新因子表，索引是股票名
    
    #净资产
    df['net_assets']=df['total_assets']-df['total_liability']
        
    #估值因子
    df_new['EP'] = df['pe_ratio'].apply(lambda x: 1/x)
    df_new['BP'] = df['pb_ratio'].apply(lambda x: 1/x)
    df_new['SP'] = df['ps_ratio'].apply(lambda x: 1/x)
    # df_new['RD'] = df['development_expenditure']/(df['market_cap']*100000000)
    df_new['CFP'] = df['pcf_ratio'].apply(lambda x: 1/x)
    
    #杠杆因子
    #对数流通市值
    # df_new['CMV'] = np.log(df['circulating_market_cap'])
    #总资产/净资产
    df_new['financial_leverage']=df['total_assets']/df['net_assets']
    #非流动负债/净资产
    df_new['debtequityratio']=df['total_non_current_liability']/df['net_assets']
    #现金比率=(货币资金+有价证券)÷流动负债
    df_new['cashratio']=df['cash_to_current_liability']
    #流动比率=流动资产/流动负债*100%
    df_new['currentratio']=df['current_ratio']
    
    #财务质量因子
    # 净利润与营业总收入之比
    df_new['NI'] = df['net_profit_to_total_operate_revenue_ttm']
    df_new['GPM'] = df['gross_income_ratio']
    df_new['ROE'] = df['roe_ttm']
    df_new['ROA'] = df['roa_ttm']
    df_new['asset_turnover'] = df['total_asset_turnover_rate']
    df_new['net_operating_cash_flow'] = df['net_operating_cash_flow_coverage']
    
    #成长因子
    df_new['Sales_G_q'] = df['operating_revenue_growth_rate']
    df_new['Profit_G_q'] = df['net_profit_growth_rate']
    # df_new['PEG'] = df['PEG']
    
    #技术指标
    df_new['RSI']=pd.Series(RSI(stocklist, date, N1=20))  
    df_new['BIAS']=pd.Series(BIAS(stocklist,date, N1=20)[0])
    df_new['PSY']=pd.Series(PSY(stocklist, date, timeperiod=20))
    
    dif,dea,macd=MACD(stocklist, date, SHORT = 10, LONG = 30, MID = 15)
    df_new['DIF']=pd.Series(dif)
    df_new['DEA']=pd.Series(dea)
    df_new['MACD']=pd.Series(macd)    
    df_new['AR'] = df['AR']    
    df_new['ARBR'] = df['ARBR']
    df_new['ATR14'] = df['ATR14']
    df_new['ARBR'] = df['ARBR']

    df_new['VOL5'] = df['VOL5']
    df_new['VOL60'] = df['VOL60']

    # 风险因子
    df_new['Skewness20'] = df['Skewness20']
    df_new['Skewness60'] = df['Skewness60']

    return df_new


def get_all_factors_dict(trade_days,q):
    original_factor_dict = {}
    for date in trade_days:
        print(date,':')
        stocklist = get_stocks_filtered(date,90,indexID=INDEX) # 获取某一天指数成分股，上市至少90天以上
        # 获取财务数据
        q_new = q.filter(valuation.code.in_(stocklist))  # 筛选条件
        q_factor = get_fundamentals(q_new,date=date)   # 获取财务数据
        q_factor.set_index('code',inplace=True)  # 将股票代码设为索引
        print(len(q_factor.index))  # 股票数量
        # 获取聚宽因子库数据
        df_jqfactor = get_df_jqfactor(stocklist,jqfactor_list,date)
        # 两表合并
        original_factor_dict[date] = get_newfactors_df(stocklist
                                                       ,pd.concat([q_factor,df_jqfactor],axis=1)
                                                       ,date)
        print("获取数据成功!")
        print('='*30)
    return original_factor_dict


# 交易日期
trade_days_all = get_trade_days_monthly(START_DATE,END_DATE)
# 分测试集
trade_days_test = trade_days_all[88:]

start = time()
original_factor_dict = get_all_factors_dict(trade_days_all,q)
end = time()
print(datetime.datetime.fromtimestamp(end-start).strftime("%M:%S:%f"))

# 保存原始数据
content = pickle.dumps(original_factor_dict) 
write_file('original_factor_dict.pkl', content, append=False)

with open('original_factor_dict.pkl','rb') as pf:
    original_factor_dict = pickle.load(pf)

