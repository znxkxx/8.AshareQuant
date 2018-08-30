# -*- coding: utf-8 -*-


#name:  MultiFactor.py
#author: XX
#create on: 20180810
# 20180824 主体框架完成. by xx
# 20180827 用pandas类型的数据改写（方便对其时间和代码）；添加注释.  by: xx
# 20180831 实现滚动预测（用linear regression model 做测试）

import time
import pandas as pd
import numpy as np

import scipy.stats as ss
import statsmodels.api as sm

import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline



class MultiFactor():
    '''
    运用机器学习方法将单因子合成多因子，参数有

    -------------------------------------------------------------------------------
    参数：
     - path_data: input数据存放路径
     - path_result_data: 结果文件保存的路径

     - file_stock_ret：个股收益率数据文件名
     - file_HS300_ret：基准回报率（HS300)文件名
     - file_Universe：股票选择范围数据 （目前为 流动性最好的2000支股票）文件名
     - file_Industry_prefix: 行业数据文件名（实际有25个文件）
     - TEST_DAYS：单因子回测区间天数 目前为1215
     - Stock_Universe: 股票池股票总数 目前为 3485
     - stock_num_per_day: 根据Univesrse筛选条件，每天入选的股票数量，目前为2000
     - factor_list：因子名称list
     - FactorShape：factor数据文件的shape[3维数组]
     - industry_num: 行业的数量，也是行业文件的数量
     - IndustryShape: 行业数据的 shape [3维数组]

     - StockReturnData: 个股收益率数据，[2维数组]
     - BenchReturnData: 基准收益率，[2维数组]
     - UniverseData：股票范围数据，[二维数组]
     - FactorData: 因子数据[3维数组]
     - IndustryData: 行业虚拟变量数据[3位数组]


     - seed: 随机数种子，用于控制随机结果
     - stock_num_per_day：每日可选股票的数量（=2000）


    -------------------------------------------------------------------------------
     输出
      - 1. 交叉验证集合上的预测准确率
          a. SVM
          b. random forest
          c. regression
          ....

      - 2. 滚动验证的准确率
          a. SVM
          b. random forest
          c. regression
          ....
    '''
    def __init__(self):

        # input file retated parameters
        # self.path_data = "D:\\quantpy\\quant_option_2\\code\\Backtest\\output\\"
        # self.path_result_data = "D:\\quantpy\\quant_option_2\\code\\Backtest\\RegTest\\"

        self.path_data = "E:/4.Project/Current/004-AshareQuant/002-data/"
        self.path_result_data = "E:/4.Project/Current/004-AshareQuant/003-result_data/"

        #self.path_data = "/Users/xinxu/Documents/8.AshareQuant/002-data/"
        #self.path_result_data = "/Users/xinxu/Documents/8.AshareQuant/003-result_data/"
        
        self.file_stock_ret = "Stock_Return.csv"
        self.file_HS300_ret = "HS300_Return.csv"
        self.file_Universe = "Universe.csv"
        self.file_Industry_prefix = "IndustryData"  # 有多个Inudstry 文件
        self.TEST_DAYS = 1215
        self.Stock_Universe = 3485

        self.stock_num_per_day = 2000
        self.factor_list = ['005', '055','057','083','088']

        #self.FactorShape = [self.TEST_DAYS,self.Stock_Universe, len(self.factor_list)]
        self.industry_num = 25
        self.IndustryShape = [self.TEST_DAYS,self.Stock_Universe, self.industry_num]


        #
        self.StockReturnSize = [self.TEST_DAYS,self.Stock_Universe]
        df_BenchReturnDataSize = [self.TEST_DAYS,1]
        df_UniverseDataSize = [self.TEST_DAYS,self.Stock_Universe]




        # machine learning related parameters
        self.seed = 42
        self.percent_select = [0.3, 0.7] # 分别代表 low_bar 和 high_bar
        self.percent_cv = 0.1
        self.method = "RandomForest"
        self.rolling_period = 100 # 滚动窗口长度为100天
        self.predict_period = 1 # 预测窗口为1天

        print("initialize finished")

    def prepare_data(self):
        '''
        这部分用于准备 需要的数据，包括
         - 个股收益率数据 StockReturnData：
         - 指数收益率数据 BenchReturnData：HS300 指数收益率，当日收盘价/昨日收盘价-1
         - 股票选取范围数据 UniverseData：目前是Liquidity 最好的 500支股票
         - 因子数据 FactorData：每支股票对应的因子数值，日期上滞后一天，t日的factor是用t-1日及以前的数据计算得到
         - 行业数据 InudstryData：Wind 2级行业分类，共计26个

        '''
        # 个股收益率数据
        # t日的收益率 = (t+1)开盘价 / t日开盘价 -1
        filename = self.path_data + self.file_stock_ret
        df_StockReturnData = pd.DataFrame(pd.read_csv(filename, index_col=['date']).stack(dropna=False)).reset_index()
        df_StockReturnData.columns = ['date','stkcd','StkRet']



        # 指数收益率数据
        filename = self.path_data + self.file_HS300_ret
        df_BenchReturnData = pd.DataFrame(pd.read_csv(filename, index_col=['date']).stack(dropna=False)).reset_index()
        df_BenchReturnData.columns = ['date','Index','HS300ret']

        # 股票选取范围数据
        filename = self.path_data + self.file_Universe
        df_UniverseData = pd.DataFrame(pd.read_csv(filename, index_col=['date']).stack(dropna=False)).reset_index()
        df_UniverseData.columns = ['date','stkcd','Universe']

        # 因子数据: 
        for i,i_factor in enumerate(self.factor_list):
            filename =self.path_data + "alpha" + i_factor + "_factor_xx.ipynb.csv"

            filedata = pd.read_csv(filename, index_col=['date'])
            filedata.drop(filedata.columns[[0]], axis=1, inplace=True)
            filedata = pd.DataFrame(filedata.stack(dropna=False)).reset_index()
            filedata.columns = ['date','stkcd','factor'+str(i)]
            if i == 0:
                df_FactorData = filedata
            else:
                df_FactorData = pd.merge(df_FactorData, filedata, on=['date','stkcd'])


        # 行业数据:
        #for i in range(self.industry_num):
            ## print("industry"+str(i))
            #filename = self.path_data + self.file_Industry_prefix + str(i) + ".csv"
            #filedata = pd.DataFrame(pd.read_csv(filename, index_col=['date']).stack(dropna=False)).reset_index()
            #filedata.columns =  ['date','stkcd','industry'+str(i)]
            #if i == 0:
                #df_IndustryData = filedata
            #else:
                #df_IndustryData = pd.merge(df_IndustryData, filedata, on=['date','stkcd'])
                
        # 将日期全部转成 date-time 格式 
        df_StockReturnData['date'] = pd.to_datetime(df_StockReturnData['date'])
        df_BenchReturnData['date'] = pd.to_datetime(df_BenchReturnData['date'])
        df_UniverseData['date'] = pd.to_datetime(df_UniverseData['date'])
        # self.Data['Universe'] = self.Data.Universe.fillna(0.0)
        # df_IndustryData['date'] = pd.to_datetime(df_IndustryData['date'])
    
        df_FactorData['date'] = pd.to_datetime(df_FactorData.date.astype(str))
        
        self.Data = pd.merge(df_StockReturnData, df_BenchReturnData, on=['date'])
        self.Data = pd.merge(self.Data, df_UniverseData, on=['date','stkcd'])
        self.Data = pd.merge(self.Data, df_FactorData, on=['date','stkcd'])
        # self.Data = pd.merge(self.Data, df_IndustryData, on=['date','stkcd'])
       
        

        print("prepare data finished")


    def factor_process(self):
        '''
        因子数据预处理
         - 横截面去极端值
         - 横截面标准化
         - 行业中性化 （todo: 加入市值中性化，即将size也纳入回归的自变量）
         -
        '''
        a = self.TEST_DAYS
        b = self.Stock_Universe
        c = len(self.factor_list)
        c1 = self.industry_num


        # 截面上 去除极端值  MAD
            
        Factor_median = self.Data.loc[:,'factor0':'factor4'].groupby(self.Data['date']).transform(lambda x: x.median()) 
        abs_diff = (abs(self.Data.loc[:,'factor0':'factor4'] - Factor_median))
        abs_diff_median = abs_diff.groupby(self.Data['date']).transform(lambda x: x.median())
        
        critical_value = Factor_median+5*abs_diff_median
        self.Data.loc[:,'factor0':'factor4'] = np.where(self.Data.loc[:,'factor0':'factor4']>critical_value,  \
                                                        critical_value, self.Data.loc[:,'factor0':'factor4'])
        
        critical_value = Factor_median - 5*abs_diff_median
        self.Data.loc[:,'factor0':'factor4'] = np.where(self.Data.loc[:,'factor0':'factor4']<critical_value,  \
                                                        critical_value, self.Data.loc[:,'factor0':'factor4'])
        
        # 标准化
        self.Data.loc[:,'factor0':'factor4'] = self.Data.loc[:,'factor0':'factor4'].groupby(self.Data['date']).transform(lambda x: (x-x.mean())/x.std()) 
        
        # 替换缺失值 to-do 
        
        # 行业中性化 to-do 
        
        # 因子序数
        
        # 因子序数化 （可选）
        
        print('因子数据处理完毕')



    # 计算 Excess Return数据
    def return_process(self):
        
        self.Data['ExcessRet'] = self.Data['StkRet'] - self.Data['HS300ret']
        # 将excess return 分为三组 [0.3, 0.7]
        # return_bin : +1: Excess Return 最高的30%; -1: Excess Return最低的 30%; 0: Excess Return位于中等的部分
        pct1 = self.percent_select[0]
        pct2 = self.percent_select[1]
        low_bar = self.Data[['date','ExcessRet']].groupby('date').transform(lambda x: x.quantile(0.3)).iloc[:,0] 
        high_bar = self.Data[['date','ExcessRet']].groupby('date').transform(lambda x: x.quantile(0.7)).iloc[:,0]
        self.Data['return_bin'] = np.where(self.Data['ExcessRet']>high_bar,1, np.where(self.Data['ExcessRet']<=low_bar,-1,0))
        
        print('Excess Return数据处理完毕')


    def construct_panel(self):

        # 构造测试数据
        # Universe:每天 Liquidity 最好的 2000支股票；每天 Excess Return 最高的 30%作为 正例， Excess Return 最低的作为 负例
        #  -训练集: 每天随机选取90%的作为训练集，其余10%作为验证集
        #    * 初步：先用全部的数据做一个训练集，选取
        #  -样本外预测：to-do
        
        
        # 定义一个 DropFlag 列，剔除：
        #  1. 因子 缺失 或者 
        #  2. return_bin 缺失 或者 
        #  3. Universe 缺失 
        
        self.Data['DropFlag'] = self.Data['Universe'].isnull() |    \
                                self.Data['return_bin'].isnull() |    \
                                self.Data.loc[:,'factor0':'factor4'].isnull().any(axis=1)
        
        print('构造Panel完成')




    def rolling_predict(self):
        
        print('进入滚动预测')
        # 保存结果在 dataframe 中 
        result_data= pd.concat([self.Data[['date','stkcd']],self.Data.loc[:,'factor0':'factor4'], self.Data['return_bin'], self.Data['DropFlag']], axis=1)
        result_data['ols'] = np.nan
        result_data['SVM'] = np.nan 
        result_data['RandForest'] = np.nan 
        # result_data.set_index(['date','stkcd'])
        
        
        # 构造一个 numpy array格式的数据 存放 当前的 日期 
        DateArray = self.Data['date'].unique()
        
        #factorname_list = "factor0"
        #for i in range(1, len(self.factor_list)):
            #factorname_list = factorname_list + ", factor" + str(i) 
        # data_full_sample = pd.concat([self.Data[['date','stkcd']],self.Data.loc[:,'factor0':'factor4'], self.Data['return_bin'], self.Data['DropFlag']], axis=1)
        #data_full_sample.set_index(['date', 'stkcd'])
        
        print('完成准备整个样本数据，进入循环')
        ii_count = 0
        for i_date in range(self.TEST_DAYS-self.rolling_period-self.predict_period):
            ii_count += 1
            print("loop" + str(ii_count))
            rolling_start_date = DateArray[i_date]
            rolling_end_date = DateArray[i_date+self.rolling_period-1]
            predict_date = DateArray[i_date+self.rolling_period]
        
            #在rolling period内构造样本 
            #条件：1.Date在 rollig period 内 
            #     2.DropFlag 是 False
            
            data_in_sample = result_data[(result_data['date']>=rolling_start_date) & \
                                              (result_data['date']<=rolling_end_date) & \
                                              ~(result_data['DropFlag'])] 
            
            # i每一期 将90%的样本划分为训练集（train)，10%的样本划分为验证集 (cross-validation) -- 尝试阶段暂时不用 
            # Xtrain, Xtest, ytrain, ytest = train_test_split(data_in_sample.loc[:,'factor0':'factor4'],data_in_sample.loc[:,'return_bin'], \
            #                                             test_size = self.percent_cv, random_state = self.seed )
            
            # 在 predict period 内构造预测数据：
            # 条件：1.Date = predict当天日期 
            #      2.DropFlag 是 False
            data_predict = result_data[(result_data.date==predict_date) & \
                                            ~(result_data.DropFlag)]
            
            if data_predict.shape[0]>0:  # 增加一个判断是否在预测当日有全部Factor=0
                # data_predict.set_index(['date', 'stkcd'])
                # 挑选predict date 最容易上涨的50只或者100支股票 等权配置资产
                #  当天开盘买入，第二天开盘卖出，对应的是当天的return  -- 数据刚好对整齐 
                
                
                Xtrain = data_in_sample.loc[:,'factor0':'factor4']
                ytrain = data_in_sample['return_bin']
                Xpredict = data_predict.loc[:,'factor0':'factor4']
                
                #Xtrain = Xtrain.dropna()
                
                #0. Lienear regression
                
                from sklearn import linear_model
                model = linear_model.LinearRegression(fit_intercept=True)
                
                #  1. SVM 测试代码
                #from sklearn import svm
                # 核函数选择 高斯函数（非线性）；惩罚系数 = 0.01
                #model = svm.SVC(kernel = "linear", C = 0.01)
                
                #print('开始估计模型')
                #time_start = time.time()
                model.fit(Xtrain, ytrain)
                #time_end = time.time()
                #print('SVM Model Fit Time：', int(time_end - time_start),'s')
                ##  2. random forest 模型
        
                #tree = DecisionTreeClassifier()
                #bag = BaggingClassifier(tree, n_estimators= 100, max_samples = 0.9, random_state = self.seed)
                #model = RandomForestClassifier(n_estimators = 100)
                #model.fit(Xtrain, ytrain)
                
                #yfit_train = model.decision_function(Xtrain)
                yfit_predict = pd.DataFrame(model.predict(Xpredict), index = Xpredict.index)
        
                # 将结果保存到文件中输出 
                result_data.ols[data_predict.index] = yfit_predict.values[:,0]
                
        return result_data 
            
            