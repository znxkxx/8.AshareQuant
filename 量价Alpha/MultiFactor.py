# -*- coding: utf-8 -*-


#name:  MultiFactor.py 
#author: XX 
#create on: 20180810
#last modified:


import pandas as pd 
import numpy as np 

import scipy.stats as ss 
import statsmodels.api as sm

import sklearn 
from sklearn import metrics 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 

from sklearn.cross_validation import train_test_split


import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# prepare data
# need to run in a single-factor test file.

'''
# 股票选取范围数据
Universe = user.Universe_one.iloc[1:,:]
Universe.insert(0,'date',user.date_range)
Universe.to_csv('D:\\quantpy\\quant_option_2\\code\\Backtest\\output\\Universe.csv', index = False)


# 个股收益率数据
StockReturn = pd.DataFrame(user.return_rate_sep[user.LOOKBACK_DAYS:,:], columns = user.Universe_one.columns)
StockReturn.insert(0,'date',user.date_range)
StockReturn.to_csv('D:\\quantpy\\quant_option_2\\code\\Backtest\\output\\Stock_Return.csv', index = False)
# filename= 'D:\\quantpy\\quant_option_2\\code\\Backtest\\output\\Stock_Return.csv'
# df_StockReturnData = pd.read_csv(filename, index_col = ['date'] )


# 指数收益率数据
BenchReturn = pd.DataFrame(user.get_index_data('000300.SH'), columns=['HS300']).pct_change().iloc[-user.TEST_DAYS:,:]
BenchReturn.insert(0,'date',user.date_range)
BenchReturn.to_csv('D:\\quantpy\\quant_option_2\\code\\Backtest\\output\\HS300_Return.csv', index = False)
# filename= 'D:\\quantpy\\quant_option_2\\code\\Backtest\\output\\HS300_Return.csv'
# df_BenchReturnData = pd.read_csv(filename, index_col = ['date'] )


# 因子数据


# 行业数据 共有25个industyr，命名为 InudstryData0.csv, IndustryData1.csv,..., InudstryData24.csv
ii = 0 
for ind in Industry.values():
    ii+=1
    df_ind = pd.DataFrame(ind[- user.TEST_DAYS:,:])
    df_ind.insert(0,'date',user.date_range)
    df_ind.to_csv('D:\\quantpy\\quant_option_2\\code\\Backtest\\output\\IndustryData' + str(ii) + '.csv', index = False)
    
    
'''

class MultipleFactor():
    '''
    运用机器学习方法将单因子合成多因子，参数有
    
    参数：
     - path_data: input数据存放路径 
     - path_result: 结果文件保存的路径
     - file_stock_ret：个股收益率数据文件名
     - file_HS300_ret：基准回报率（HS300)文件名
     - file_Universe：股票选择范围数据 （目前为 流动性最好的2000支股票）文件名
     - file_Industry_prefix: 行业数据问价名（实际有25个文件）
     - seed: 随机数种子，用于控制随机结果 
     - stock_num_per_day：每日可选股票的数量（=2000）
     - file_factor_list：因子名称list
     - 因子名称：list 
    '''
    def __init__(self):
        
        # input file retated parameters
        self.path_data = "D:\\quantpy\\quant_option_2\\code\\Backtest\\output\\"
        self.path_result = "D:\\quantpy\\quant_option_2\\code\\Backtest\\RegTest\\"
        self.file_stock_ret = "Stock_Return.csv"
        self.file_HS300_ret = "HS300_Return.csv"
        self.file_Universe = "Universe.csv"
        self.file_Industry_prefix = "IndustryData"  # 有多个Inudstry 文件 
        self.TEST_DAYS = 1215
        self.Stock_Universe = 3485
        
        self.stock_num_per_day = 2000
        self.file_factor_list = ['005', '055','057','083','100']
        
        self.FactorShape = [self.TEST_DAYS,self.Stock_Universe, len(self.file_factor_list)]  
        self.industry_num = 25
        self.IndustrySize = [self.TEST_DAYS,self.Stock_Universe, self.industry_num]
        
        
        # 
        self.StockReturnSize = [self.TEST_DAYS,self.Stock_Universe]
        self.BenchReturnDataSize = [self.TEST_DAYS,1]
        self.UniverseDataSize = [self.TEST_DAYS,self.Stock_Universe]

        
        self.StockReturnData = np.empty(self.StockReturnSize)
        self.BenchReturnData = np.empty(self.BenchReturnDataSize)
        self.UniverseData = np.empty(self.UniverseDataSize)
        self.FactorData = np.empty(self.FactorShape)
        self.IndustryData = np.empty(self.IndustrySize)
        
        self.StdFactor = self.FactorData.copy()
        self.ResStdFactor = self.StdFactor.copy()
        self.RankResStdFactor = self.StdFactor.copy() 
        

        # machine learning related parameters
        self.seed = 42        
        self.percent_select = [0.2, 0.2] # 分别代表 low_bar 和 high_bar
        self.percent_cv = 0.1
        self.method = "RandomForest"
        self.rolling_period = 250 # 滚动窗口长度为250天
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
        filename = self.path_data + self.file_stock_ret
        df_StockReturnData = pd.read_csv(filename, index_col = ['date'] ) # t日的收益率 = (t+1)开盘价 / t日开盘价 -1 
        self.StockReturnData = df_StockReturnData.values  
        
        # 指数收益率数据
        filename = self.path_data + self.file_HS300_ret
        df_BenchReturnData = pd.read_csv(filename, index_col = ['date'] )
        self.BenchReturnData = df_BenchReturnData.values.reshape(self.BenchReturnDataSize)    
        
        
        # 股票选取范围数据
        filename = self.path_data + self.file_Universe
        df_UniverseData = pd.read_csv(filename, index_col = ['date'] )
        self.UniverseData = df_UniverseData.values
        
        
        # 因子数据
        for i,i_factor in enumerate(self.file_factor_list):
            filename =self.path_data + "alpha" + i_factor + "_factor_xx.ipynb.csv"
            
            df_FactorData = pd.read_csv(filename, index_col = ['date'] )
            df_FactorData.drop(df_FactorData.columns[[0]], axis=1, inplace=True)
            self.FactorData[:,:,i] = df_FactorData.values
            
          
        # 行业数据
        filename = self.path_data + self.file_Industry_prefix + str(0) + ".csv"
        df_IndustryData = pd.read_csv(filename, index_col = ['date'] )
        for i in range(1,self.industry_num):
            # print("industry"+str(i))
            filename = self.path_data + self.file_Industry_prefix + str(i) + ".csv"
            df_IndustryData = pd.read_csv(filename, index_col = ['date'] )
            self.IndustryData[:,:,i] = df_IndustryData.values
            
        print("prepare data finished")
        

        
        
    def factor_process(self):
        '''
        因子数据预处理
         - 横截面去极端值
         - 横截面标准化 
         - 行业中性化 （todo: 加入市值中性化，即将size也纳入回归的自变量）
         - 
        ''' 
        a, b, c = self.FactorData.shape
        a1, b1, c1 = self.IndustryData.shape
        
               
        self.StdFactor = self.FactorData.copy()
        for factor_i in range(c):
            # print("factor_i=", factor_i)
            for date_i in range(a): 
            
                Factor_median = np.nanmedian(self.FactorData[date_i,:,factor_i])
                abs_diff_Factor_median = np.nanmedian(np.abs(self.FactorData[date_i,:,factor_i]-Factor_median))
                
                self.StdFactor[date_i,:,factor_i][self.StdFactor[date_i,:,factor_i]>(Factor_median+5*abs_diff_Factor_median)] = \
                    Factor_median+5*abs_diff_Factor_median
                self.StdFactor[date_i,:,factor_i][self.StdFactor[date_i,:,factor_i]<(Factor_median-5*abs_diff_Factor_median)] = \
                    Factor_median-5*abs_diff_Factor_median
         
        
        
            
            # 沿 dimention2 （stock) 做标准化 
            
            # 缺失处理如果某一天 某个 N个因子中有k (0<k<N)个因子为missing，
            #  将该因子当天所有样本股票因子的均值替换
            #  --相当于标准化之后取值为0
            # todo：改为行业均值而非整体均值 
            self.StdFactor[:,:, factor_i] = (self.StdFactor[:,:, factor_i] - np.nanmean(self.StdFactor[:,:, factor_i])) / \
                np.nanstd(self.StdFactor[:,:, factor_i], axis=1, keepdims=True)
            
            # 判断条件 当天 空因子的数量 大于0 且小于 factor number
            
            cond = ( np.sum(np.isnan(self.StdFactor), axis=2, keepdims=True)>0 ) & \
                   ( np.sum(np.isnan(self.StdFactor), axis=2, keepdims=True)<c)  # c = factor number 
            # cond: 代表存在 部分空 factor的取值的情况  
            #　np.sum(cond * np.isnan(self.StdFactor))
            self.StdFactor[cond * np.isnan(self.StdFactor)] = 0  
            
        print("factor process finished")
        
        
        # 行业中性化 对行业dummy做regression 取残差序列 
        # 缺失值处理：取所有公司的均值替换（todo: 用行业均值）
        #  - 特殊处理：部分情况存在全部股票的factor数值都为缺失值--》定义对应的 residual factor 也为 缺失值
        for i_factor in range(c):  # 针对 factor-date循环 
            # print("factor=", i_factor)
            for i_date in range(a):
                y = self.StdFactor[i_date,:,i_factor]
                x = np.empty((b1,c1))  # x是 industry dummies 
                for i_industry in range(c1):
                    x[:,i_industry] = self.IndustryData[i_date,:,i_industry]
                
                
                mask = (np.isnan(y))
                x = x[~mask,:]
                y = y[~mask]  
                if y.size == 0:  # 如果某一天所有的 factor 都是 missing
                    self.ResStdFactor[i_date,:,i_factor] = np.nan
                else:
                    est = sm.OLS(y,x).fit()
                    self.ResStdFactor[i_date,:,i_factor][~mask] = est.resid 
        
        self.ResStdFactor[np.isnan(self.StdFactor)] = np.nan
        # 对部分 factor 为空的情况做处理             
        cond = ( np.sum(np.isnan(self.ResStdFactor), axis=2, keepdims=True)>0 ) & \
               ( np.sum(np.isnan(self.ResStdFactor), axis=2, keepdims=True)<c)  # c = factor number 
        # cond: 代表存在 部分空 factor的取值的情况  
        #　np.sum(cond * np.isnan(self.ResStdFactor))
        self.ResStdFactor[cond * np.isnan(self.ResStdFactor)] = 0  # 理论上这一步处理之后痐出现 某一天 部分因子为missing的特殊情况 
                    
        print("factor 行业中心化 finished")
        
        
        for i_factor in range(c):           
            df = pd.DataFrame(self.ResStdFactor[:,:,i_factor].reshape(b,a))
            self.RankResStdFactor[:,:,i_factor] = df.rank().values.reshape(a,b) / \
                (np.sum(1-np.isnan(self.ResStdFactor[:,:,i_factor]),axis=1, keepdims=True))
        
        self.RankResStdFactor[np.isnan(self.ResStdFactor)] = np.nan   

        print("factor 序数化 finished")
        
        
    # 计算 Excess Return数据 
    def return_process(self):
        
        self.ExcessRet = np.row_stack((self.StockReturnData[1:,]-self.BenchReturnData[1:,:],np.zeros((1,self.StockReturnData.shape[1]))))
        self.ExcessRet[-1] = np.nan
      
    
    # 构造测试数据 
    # Universe:每天 Liquidity 最好的 500支股票；每天 Excess Return 最高的 30%作为 正例， Excess Return 最低的作为 负例
    #  -训练集: 每天随机选取90%的作为训练集，其余10%作为验证集 
    #    * 初步：先用全部的数据做一个训练集，选取
    #    * 后续：滚动测试，5年作为，每月滚动一次（这样训练集的差别可能不会很大） 总体有6年，可以做12个测试 
    #  -样本外预测：to-do 
    
    
    def construct_panel(self, data_in, data_out):
        '''
        将二维或者三维数据转换为 panel
        输入：2维或者3维数组
        返回：2维数组
        '''
        if len(data_in.shape) == 2:
            # data_out = np.empty((data_in.shape[0]*data_in.shape[1],1))
            for i_count in range(data_in.shape[0]):
                low_bar = i_count * data_in.shape[1]
                high_bar = (i_count+1) * data_in.shape[1]
                data_out[low_bar:high_bar,0] = data_in[i_count,:]
                
        elif len(data_in.shape) == 3:
            # data_out = np.empty((data_in.shape[0]*data_in.shape[1],data_in.shape[2]))  
            for i_factor in range(data_in.shape[2]):  
                for i_date in range(data_in.shape[0]):
                    low_bar = (i_date)* data_in.shape[1]
                    high_bar = (i_date+1)* data_in.shape[1]
                    data_out[low_bar:high_bar, i_factor] = data_in[i_date, :, i_factor]
                 
        else:    
            raise "error, can only processing arrays not in 2-dimention or 3-dimention!"
        
        return data_out 
    
    
    def full_sample_test(self):
        '''
        基于整体样本构建训练集和交叉验证集合
        
        '''

        # 生成 return 二元判断指标：excess return最高的30%为1； excess return最低的为0；中间部分为nan.--后续需要剔除

        self.return_bin = np.empty(self.StockReturnData.shape)
        rank = np.argsort(-(self.StockReturnData-self.BenchReturnData)* self.UniverseData, axis=1)  # 在横截面上按照excess return的排列，return越大，rank越大
        self.return_bin[rank<self.percent_select[1]*self.stock_num_per_day ] = 3 # 排序在前面的 excess return 较高
        self.return_bin[rank>=((1-self.percent_select[0])*self.stock_num_per_day)] = 1 # 排序在后面的 excess return 较低
        self.return_bin[(rank<((1-self.percent_select[0])*self.stock_num_per_day)) * (rank>=self.percent_select[1]*self.stock_num_per_day)] = 2
        self.return_bin[rank>self.stock_num_per_day] = np.nan # 调整 missing value 的rank 

        # 保留所有
        # y 是 excess return数据 -- 标签
        self.y_in_sample = np.multiply(self.return_bin,self.UniverseData)
        # x 是 处理之后的factor数据 -- 特征 
        self.x_in_sample = np.multiply(self.RankResStdFactor, self.UniverseData.reshape( \
            self.RankResStdFactor.shape[0],self.RankResStdFactor.shape[1],1))

        # 将数据转换为二维格式 
        # 按照 股票-日期 作为 panel data 
        
        #  - self.y_in_sample 转为 1列，(excess return:bin)
        #  - self.x_in_sample 转为 5列 （5个factor）
        
        self.y_in_sample_panel = np.empty((self.y_in_sample.shape[0]*self.y_in_sample.shape[1],1))
        self.y_in_sample_panel = self.construct_panel(self.y_in_sample, self.y_in_sample_panel)
        
        #for i_count in range(self.y_in_sample.shape[0]):
            #low_bar = i_count * self.y_in_sample.shape[1]
            #high_bar = (i_count+1) * self.y_in_sample.shape[1]
            #self.y_in_sample_panel[low_bar:high_bar,0] = self.y_in_sample[i_count,:]

        self.x_in_sample_panel = np.empty((self.x_in_sample.shape[0]*self.x_in_sample.shape[1],self.x_in_sample.shape[2]))  
        self.x_in_sample_panel = self.construct_panel(self.x_in_sample, self.x_in_sample_panel)
        
        #for i_factor in range(self.x_in_sample.shape[2]):  
            #for i_date in range(self.x_in_sample.shape[0]):
                #low_bar = (i_date)* self.x_in_sample.shape[1]
                #high_bar = (i_date+1)* self.x_in_sample.shape[1]
                #self.x_in_sample_panel[low_bar:high_bar, i_factor] = self.x_in_sample[i_date, :, i_factor]

        
        # 剔除所有 x 都为misisng的value
        #  理论上这里应该是所有的x全部为missing或者全部非missing，
        #  需要进一步检查 
        mask = np.all(np.isnan(self.x_in_sample_panel), axis = 1)
        mask = mask | np.isnan(self.y_in_sample_panel)[:,0]
        mask = ~mask
    
        list_mask = mask.tolist()
        
        y_in_sample_panel = self.y_in_sample_panel[mask]  
        x_in_sample_panel = self.x_in_sample_panel[list_mask,:]
        

        Xtrain, Xtest, ytrain, ytest = train_test_split(x_in_sample_panel, y_in_sample_panel, \
                                                        test_size = test.percent_cv, random_state = self.seed )
        
        # 训练样本                
        # 先用 SVM 测试代码 
        if self.method == "SVM":
            from sklearn import svm 
            # 核函数选择 高斯函数（非线性）；惩罚系数 = 0.1
            model = svm.SVC(kernel = "rbf", C = 0.1) 

            model.fit(Xtrain, ytrain)
        # random forest 模型      
        elif self.method == "RandomForest":
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import BaggingClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import BaggingClassifier
            from sklearn.ensemble import RandomForestClassifier
            tree = DecisionTreeClassifier()
            bag = BaggingClassifier(tree, n_estimators= 100, max_samples = 0.9, random_state = test.seed)            
            model = RandomForestClassifier(n_estimators = 100)
            model.fit(Xtrain, ytrain)
            
            
        yfit = model.predict(Xtest) # 在交叉验证集上的预测值 

        # 交叉验证集合的结果
        from sklearn import metrics 

        print("交叉验证集预测结果如下:")
        
        print(classification_report(ytest, yfit))
        print("\n"*1)
        mat = confusion_matrix(ytrain, ypred_train)
        sns.heatmap(mat.T, square=True, annot=True, fmt= 'd', cbar=False)
        plt.xlabel('true value')
        plt.ylabel('predicted value')        
        # plt.close
        
        

