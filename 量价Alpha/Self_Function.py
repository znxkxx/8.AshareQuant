# -*- coding: utf-8 -*-


from Execution import Execution

import time
import numpy as np
import pandas as pd
import scipy.stats as ss 
import statsmodels.api as sm

#%%
class UserWrite(Execution):
    '''
    继承自Excution, 用户设置回测环境、获取数据、输入策略的类
    用户在UserWrite中书写函数, 然后直接执行即可
    
    用户书写 Functions:
        __init__:       设置回测环境、回测参数
        fetch_data:     选取数据
        Generate:       生成Alpha的函数
        '''

    def __init__(self,Date_start_test='2017-01-01', Date_end_test = '2018-01-01', \
           Date_start_backtest = '2017-01-01', Date_end_backtest = '2018-01-01',\
           test_type='test', freq='m', BENCHMARK = "000300.SH",\
           FACTOR_NAME = 'alpha101', STRATEGY_NAME = None,\
           industry_type = ('Wind',2),\
           UNIVERSE_CONDITION = (None, 'liquidity',2000, None), LOOKBACK_DAYS = None,\
           DECAY = 4,DELAY = 0, NEUTRALIZATION = "Market", MAX_STOCK_WEIGHT = 0.05,\
           BUY_SELL_TYPE = "open"):

        
        Execution.__init__(self,Date_start_test, Date_end_test,Date_start_backtest,\
         Date_end_backtest,test_type, freq, BENCHMARK,\
         FACTOR_NAME, STRATEGY_NAME,industry_type)
        
        self.update_paras(UNIVERSE_CONDITION,LOOKBACK_DAYS,DECAY,DELAY,\
           NEUTRALIZATION, MAX_STOCK_WEIGHT,BUY_SELL_TYPE)
        
        
    #--------------------------------------------------------------------------------
    # Step1. 获取需要的基础数据，作为class attribute存储
    def fetch_data(self):
        '''
        用户在该函数内输入需要获取的数据，通过父类prepare函数调用运行。
        '''  
        self.ClosePrice = self.get_trade_data('close')                           
        self.OpenPrice = self.get_trade_data('open')     
        
    # ---------------------------------------------------------------------------------
    
    # -------------------------------------------
    # function 1. rank()
    def _rank(self, data, asce = False):
        '''
        对应Formularic 101 Alpha中的rank(x)函数
        功能：给出所有股票在每个时点上的排序，并且用当天的股票总数做一个调整，使用pandas的rank()函数计算
        返回：与原来属性 attrname形状相同的ndarray 矩阵
        '''
        # step1 转换为pandas
        df_data = pd.DataFrame(data)        
        # step2 pd.rank()计算每个截面时点上每只股票的排序(总数标准化，降序排列，越小排名对应的数值越大）         
        cross_rank = df_data.rank(axis = 1, ascending = asce)
        cross_rank = (cross_rank.values.T/cross_rank.max(axis = 1).values.T).T
        return cross_rank    

        
    # -------------------------------------------
    # function 2: delay(x,d)
    # def _delay(self, data, d):
      #  '''
      #  这个函数直接用切片方式获取
      #  '''
      #  return data[:-d,:]
    # -------------------------------------------


    
    # -------------------------------------------
    # function 3: correlation()
    def _correlation(self, data1, data2):
        '''
        返回属性 data1和data2在过去d天内的correlation。
        d由数据的行数决定
        输入为numpy格式的矩阵，返回值为1XN的矩阵
        '''     
        corr = np.full((1,data1.shape[1]),np.nan) # 定义一个全nan的矩阵
        for postion_i in range(data1.shape[1]):
            corr[0,postion_i] = np.corrcoef(data1[:,postion_i],data2[:,postion_i])[0,1] 
        return corr 
    # -------------------------------------------
    
    
    # -------------------------------------------       
    # function 4: covariance()
    def _covariance(self, data1, data2):
        '''
        !!! 待测试
        返回属性 data1和data2在过去d天内的covariance。
        d由数据的行数决定
        输入为numpy格式的矩阵，返回值为1XN的矩阵
        '''     
        corr = self._correlation(data1, data2) # 定义一个全nan的矩阵
        
        return np.nanstd(data1, axis = 0, keepdims = True)*corr* np.nanstd(data2, axis = 0, keepdims = True)
        
    # -------------------------------------------

    
    # -------------------------------------------    
    # function 5: scale(x,a=1)
    def _scale(self, data, base=1):
        '''
        by: xx
        Last modify: 
        ***功能：在每个时间点上，计算cross-sectional上计算将x进行归一化: sum(abs(data) = base)       
        ***输入：
          1. x为股票数据，T x N 维矩阵，numpy ndarray格式
          2. base，整数 默认取值为1    
        ***输出：
          1. 归一化之后的x，仍是 T x N 维矩阵，numpy ndarray格式
        
        TODO:
          * 增加错误检查和报错机制
          '''
        return data/np.nansum(np.abs(data),axis = 1, keepdims = True)*base
    # -------------------------------------------

    
    # -------------------------------------------    
    # function 6：delta(x,d)
    def _delta(self, data):
        '''
        by: xx
        Last modify: 2018-06-18
        ***功能： 计算 今日的value相对于d日以前value的变动,d由data的行数确定: d = data.shape[0]
        ***输入：
          1. data - 股票在某个日期范围内的某个指标， T X N维 矩阵， numpy ndarray格式
        ***输出：
          1. 1 X N 维度矩阵， numpy ndarray格式

        TODO:
          * 
          '''
        return data[-1:,:]-data[0:1,:]
    # -------------------------------------------  

    
    # -------------------------------------------
    # function 7:signedpower(x,a)
    # def _signedpower(self, data, a):
    #    return np.power(data, a)

    # -------------------------------------------
     
    
    # -------------------------------------------
    # function 8: decaylinear(x,d)
    def _decay_linear(self, data):
        '''
        by: xx
        Last modify: 2018-06-18
        ***功能： 计算今日value在过去d日的加权移动平均值，权重由距离今天的天数(包括今天）确定，并且rescale为1
                 如果某一天有缺失，用前一个交易日的数据补充
        ***输入：
          1. data - 股票在某个日期范围内的某个指标， T X N维 矩阵， numpy ndarray格式
          1'. d: 隐含，由data的行数T确定
        ***输出：
          1. 1 X N 维度矩阵， numpy ndarray格式

        TODO:
          * 提高效率，目前是循环，后续考虑怎么采用矩阵计算公式
          * 增加missing value的处理，如果过去d日内的missing超过2/3，就设定alpha = 1
          '''    
        # pass
        # 计算weight（d x 1）
        df_data = pd.DataFrame(data)
        df_data = df_data.fillna(method = 'ffill')
        
        
        weight = np.zeros((data.shape[0],1))
        for i in range(data.shape[0]):
            weight[i] = (1+i)/((data.shape[0]*(data.shape[0]+1))/2)
            
        return np.nansum(df_data.values* weight, axis = 0, keepdims = True)
    # -------------------------------------------
    
    
    # -------------------------------------------        
    # function 9: indneutralize(x,g)
    def _industry_neutral(self, data, Industry):
        '''
        功能：返回指标减行业均值之后的数值
        输入：data-     TxN矩阵，某一个待选指标
             Industry- 行业信息数据，3维(TxNxInd) 0-1矩阵，第3维度是行业
        输出：TxN矩阵
        
        '''
        demean_data = data.copy()
        for ii in range(Industry.shape[2]):
            ind_i = Industry[:,:,ii].reshape(Industry.shape[0], Industry.shape[1]) # 截取需要计算日期范围内的 0-1 行业息息
            ind_i[np.isnan(data)] = 0 # 进一步判断 如果对应的数据是 missing 该公司当天对应的权重也设定为0 
            weight = ind_i / np.nansum(ind_i, axis=1, keepdims=True) # 计算出行业内每只股票的权重
            demean = np.where(ind_i>0,data - np.nansum(weight * data, axis=1, keepdims=True), 0) # 计算de-mean之后的数值
            demean_data = np.where(ind_i>0, demean, demean_data)
            
        return demean_data  

    # -------------------------------------------    
    

    # -------------------------------------------
    # function 10： ts_min(x,d)
    def _ts_min(self, data):
        '''
        by: xx
        Last modify: 2018-06-18
        ***功能： 范围数据在过去d日内的最小值，d内涵给出
        ***输入：
          1. data - 股票在某个日期范围内的某个指标， T X N维 矩阵， numpy ndarray格式
          2. d，由data.shape[0]确定
        ***输出：
          1. 1 X N 维度矩阵， numpy ndarray格式

        TODO:
          * 
          '''              
        return np.nanmin(data,axis = 0, keepdims = True)
    # -------------------------------------------
    

    # -------------------------------------------
    # function 11:
    
    def _ts_max(self, data):
        '''
        by: xx
        Last modify: 2018-06-18
        ***功能： 范围数据在过去d日内的最小值，d内涵给出
        ***输入：
          1. data - 股票在某个日期范围内的某个指标， T X N维 矩阵， numpy ndarray格式
          2. d，由data.shape[0]确定
        ***输出：
          1. 1 X N 维度矩阵， numpy ndarray格式

        TODO:
          * 
          '''          
        return np.nanmax(data, axis = 0, keepdims = True)
    # -------------------------------------------   
    
    
    # -------------------------------------------
    # function 12: ts_argmin(x,d)
    def _ts_argmax(self, data):
        '''
        价格在过去d日内最大值发生的日期--改为距离今天的日期
        data为dXN矩阵
        返回值为1XN矩阵
        '''
        #如果一列全部为nan则无法使用nanargmax函数，故在第一行前添加一行数字
        #生成一行最小值-1的数组，并将该数组放在data的前一行，拼接成新的数组data_arrange
        data_append = np.full((1,data.shape[1]),np.nanmin(data)-1)
        data_arrange = np.concatenate([data_append,data])
        #对data_arrange进行nanargmax，选取除nan外的最大值
        argmax = np.nanargmax(data_arrange,axis = 0)
        return np.where(argmax ==0,np.nan,data.shape[0]-argmax).reshape(1,argmax.size)   
    # -------------------------------------------
    
    
    # -------------------------------------------
    # function 13: ts_argmin(x,d)
    
    def _ts_argmin(self, data):
        '''
        价格在过去d日内最大值发生的日期--改为距离今天的日期
        data为dXN矩阵
        返回值为1XN矩阵
        '''
        #如果一列全部为nan则无法使用nanargmax函数，故在第一行前添加一行数字
        #生成一行最小值-1的数组，并将该数组放在data的前一行，拼接成新的数组data_arrange
        data_append = np.full((1,data.shape[1]),np.nanmax(data)+1)
        data_arrange = np.concatenate([data_append,data])
        #对data_arrange进行nanargmax，选取除nan外的最大值
        argmin = np.nanargmin(data_arrange,axis = 0)
        return np.where(argmin ==0,np.nan,data.shape[0]-argmin).reshape(1,argmin.size)     
    # -------------------------------------------
    
    
    # -------------------------------------------    
    # function 14: ts_rank
    def _ts_rank(self , data):
        '''
        对应Formularic 101 Alpha中的ts_rank(x)函数
        功能：给出每只股票在过去d天内的排名，并且用总天数做一个调整，使用pandas的rank()函数计算
        返回：与原来属性 attrname形状相同的ndarray 矩阵
          如果一只股票过去一段时间都在停牌（或者d天内有一半以上停盘）-->剔除

        TO-DO: 增加 missing value 容错处理
        '''
        df_data = pd.DataFrame(data)
        df_data_rank = df_data.rank(axis=0 , ascending=False)
        # pd.rank()计算每个截面时点上每只股票的排序(总数标准化，降序排列，排名数字小，对应数值越大）
        
        T = data.shape[0]
        ts_rank = (((df_data_rank.values) / np.nanmax((df_data_rank.values) , axis=0))[T-1:T , :]) 
        # 注意 Slicing返回1个 1xN的矩阵
        return ts_rank        
    # -------------------------------------------

    
    # -------------------------------------------    
    # function 15: sum(x,d)
    
    def _ts_sum(self, data):
        '''
        主要是为了调整过去d日有停牌的情况 np.nansum()求和不能调整天数差异的影响
        '''
        # 计算非空的天数
        non_missing_days = data.shape[0] - np.sum(np.isnan(data), axis = 0, keepdims = True)
        return np.sum(data, axis = 0, keepdims = True) * data.shape[0] / non_missing_days
    # -------------------------------------------
    
    
    # ------------------------------------------- 
    # function 16: 行业去中心化
    def _industry_neutral(self, Industry, data, start, end):
        '''
        功能：返回指标减行业均值之后的数值
        输入：data-     TxN矩阵，某一个待选指标
             Industry- 行业信息数据，3维(TxNxInd) 0-1矩阵，第3维度是行业
        输出：TxN矩阵
        
        '''
        for ind_i in Industry.values():
            ind_i = ind_i[start:end+1, :, :] # 截取需要计算日期范围内的 0-1 行业息息
            ind_i[np.isnan(data)] = 0 # 进一步判断 如果对应的数据是 missing 该公司当天对应的权重也设定为0 
            weight = ind_i / np.nansum(ind_i, axis=1, keepdims=True) # 计算出行业内每只股票的权重
            demean = np.where(ind_i>0,data - np.nansum(weight * data, axis=1, keepdims=True), 0) # 计算de-mean之后的数值
            demean_data = np.where(ind_i>0, demean, demean_data)
            
        return demean_data
    # -------------------------------------------


    # ------------------------------------------- 
    # function 17: sma(a,m,n) - 计算移动平均
    def _SMA(self, data, m, n):
        '''
        实现国泰君安Alpha中的SMA()函数
        输入：
        输出：最新一期的 移动平均数值
        '''
        
        sma = np.zeros((data.shape))
        beta = m/n 
        for i in range(data.shape[0]):
            if i == 0:
                sma[i] = (1 - beta) * data[i]
            else:
                sma[i] = beta * sma[i-1] + (1 - beta) * data[i]
                
        return (sma[data.shape[0]-1:data.shape[0],:])
    # -------------------------------------------  
      
        
    # -------------------------------------------   
    # function 18:  
    def _DTM(self, High, Open):
        '''
        OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
        input: T x N维 数组， numpy格式
        output: T-1 x N 维 数组，numpy格式 
        '''
        DTM_max = ( (High-Open)[1:,:] > np.diff(Open, axis =0) ) * (High-Open)[1:,:] + \
                  ( (High-Open)[1:,:] <= np.diff(Open, axis =0) ) * np.diff(Open, axis =0) 
        return (np.diff(Open, axis=0) >0 ) * DTM_max    
    # -------------------------------------------
        
        
    # -------------------------------------------  
    # function 19                                  
    def _DBM(self, Low, Open):
        '''
        OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
        input: T x N维 数组， numpy格式
        output: T-1 x N 维 数组，numpy格式 
        '''
        DBM_max = ( (Open-Low)[1:,] > np.diff(Open, axis =0) ) * (Open-Low)[1:,:] + \
                  ( (Open-Low)[1:,] <= np.diff(Open, axis =0) ) * np.diff(Open, axis=0)
        return (np.diff(Open, axis=0) < 0 ) * DBM_max
    # -------------------------------------------
    
    
    # -------------------------------------------    
    # function 20
    def _TR(self, High, Low, Close) :
        
        '''
        _TR(self, High, Low, Close) = MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)) )
        input: T x N维 数组， numpy格式
          *High - 最高价
          *Low - 最低价 
          *Close - 收盘价
        output: T-1 x N 维 数组，numpy格式 
        '''

        TR_ABS1 = np.abs(High[1:,:]-np.diff(Close, axis=0)) 
        TR_ABS2 = np.abs(Low[1:,:] - np.diff(Close, axis=0))

        TR_MAX1 = np.where((High-Low)[1:,:] > TR_ABS1, (High-Low)[1:,:], TR_ABS1)
        TR_MAX2 = np.where(TR_MAX1 > TR_ABS2, TR_MAX1, TR_ABS2)
        return TR_MAX2    
    # -------------------------------------------
    
    
    # -------------------------------------------     
    # function 21   
    def _SIGN(self, data):
        '''
        返回 data 矩阵中每一个元素的符号
        input - T x N维 数组 
        output - T x N维 数组
        '''
        return ((data>0) * 1 + (data==0) * 0 + (data<0) * -1) 
    # -------------------------------------------
    
    
    # ------------------------------------------- 
    # function 22
    def _REGBETA(self, Y, X):
        '''
        如果Y和X都是个股指标，函数为 Y和X的每一列对应做回归；如果Y是市场指标（只有1列），函数为Y和X的每一列做回归
        Y 对 X 的回归的系数 仅限一元回归模型
        input: numpy 数组， 每一行代表一个观测点，每一列代表一个股票 
          Y -  数组 TxN 或者 Tx1, Tx1代表的是像 market returns这样的指标 
          X -  多维数组  TxN 
         需要将每一列的Y对X进行回归，得到对应的Beta系数
        output:  1xN 的 二维数组
        
        将每一列的y和每一列的X回归，beta系数作为数组输出
        '''   
        beta = np.zeros((1,X.shape[1]))

        for i in range(X.shape[1]):
            if Y.shape[1] == 1:
                y = Y
            else: 
                y = Y[:,i:i+1]     

            x = np.concatenate((np.ones((X.shape[0], 1)), X[:,i:i+1]), axis=1)

            try:    
                est = sm.OLS(y,x).fit()
                beta[0,i] = est.params[1]
            except:
                beta[0,i] = np.nan

        return beta
    # -------------------------------------------
    
    
    # -------------------------------------------
    # function 23
    def _REGRESI(self, Y, X):
        '''
        如果Y和X都是个股指标，函数为 Y和X的每一列对应做回归；如果Y是市场指标（只有1列），函数为Y和X的每一列做回归
        Y 对 X 的回归的残差 仅限一元回归模型
        input: numpy 数组， 每一行代表一个观测点，每一列代表一个股票
          Y - 数组 TxN 或者 Tx1, Tx1代表的是像 market returns这样的指标
          X - 多维数组  TxN 
         其中 T代表日期，N代表股票个数 
        output:  TxN 的数组
        
        将每一列的y和每一列的X回归，残差序列作为数组输出
        '''   
        resi = np.zeros((X.shape[0],X.shape[1]))

        for i in range(X.shape[1]):
            if Y.shape[1] == 1:
                y = Y
            else: 
                y = Y[:,i:i+1]     

            x = np.concatenate((np.ones((X.shape[0], 1)), X[:,i:i+1]), axis=1)

            try:    
                est = sm.OLS(y,x).fit()
                resi[:,i] = est.resid
            except:
                resi[:,i] = np.nan

        return resi
    # -------------------------------------------
        
    # ==========================================================================================
    

    def Generate(self, i, di, alpha):
        '''
        用户输入自定义策略，通过父类calculate函数计算alpha矩阵。
        
        参数说明: 
        -------
        di-可以理解为当前日期的指针，i-代表的是当前日期所在的index位置（用于在Universe_one中定位，Universe_one是一个0/1的DataFrame矩阵
          代表某一个时点上每一个股票是否正常交易
        
          '''
        
        
        return alpha


#%% 运行
if __name__ == '__main__':
    '''
    主函数
    '''
    print('程序运行开始：')
    time_start = time.time()
    
    user = UserWrite()
    user.data_prepare()
#     user.run_program()

    # 需要将当前文件名录入
#     user.save_profile('alpha_test_function_xx.ipynb')

time_end = time.time()
print('程序运行结束！')
print('运行时间为：', int(time_end - time_start),'s')


    # 