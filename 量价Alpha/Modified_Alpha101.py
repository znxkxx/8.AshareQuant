#Alpha Computation

import numpy as np
import pandas as pd 


def modified_alpha1():
    
    # ################################################################################
    # alpha1 = (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    # 含义：买卖策略

    # rank 为按照降序排列
    # ts_AgMax 最大值距离今天的天数 -- 买入的是 rank值较大--ts_argmax较小，也就是最近发生了 下述现象： 当日下跌且过去20天的波动较大 或者  当日上涨且收盘价较高 
    # 理论上看的是 股票价格的二阶矩； 下跌用二阶中心距，上涨用二阶原点矩 由于是在日度高频上，可以认为 return 服从 random walk, E(return) = 0 ，两种Moments是等价的
    
    # 逻辑：
    # return > 0 ? --  1. Yes >0 看一周内的价格走势--      a) 近期价格最高点 -- 高权重买入    （3天内Momentum, 3-5天 reversal）
    #                                                   b) 最高价在远期   -- 高权重卖出 
    #
    #                  2. No  <0 看一周内return的波动 --- c) 近期高波动 -- 价格反转 买入
    #                                                   d) 近期低波动 -- 价格momentum 卖出 （基于vol判断momentum 或者 reversal)

    # 回测结果：15年之前稳健增长；15年牛市暴涨暴跌；16年相对平稳，17年下降趋势
    ##cum_return_rate           1.41372
    ##final_return_rate        0.200348
    ##beta                     0.961468
    ##alpha                    0.100939
    ##sharpe ratio             0.432683
    ##information ratio        0.328493
    ##turnover rate            0.143955
    ##max drawdown             0.496233
        ##drawdown start date  2015-09-15
        ##drawdown end date          None
    ##fitness                  0.510445
    
    #
    # 改1：策略中价格为非标准指标，容易受高股价或者低价公司（尤其是复权之后）的极端影响 改为收盘价除以前面4天的收盘均价
    
    # ################################################################################
    DELAY = self.DELAY
    d = 5
    d_r = 20
    
    #alpha_01closeadj：(d+1)XN
    #alpha_01return：(d+d_r+1)XN
    #最后一行为今天的数据，没用
    
    alpha01_closeadj = self.ClosePrice[di-DELAY-d:di-DELAY+1,:]*self.adjfactor[di-DELAY-d:di-DELAY+1,:]
    
    alpha01_return = (self.ClosePrice[di-DELAY-(d+d_r):di-DELAY+1,:]*self.adjfactor[di-DELAY-(d+d_r):di-DELAY+1,:]) - \
    (self.ClosePrice[di-DELAY-(d+d_r)-1:di-DELAY,:]*self.adjfactor[di-DELAY-(d+d_r)-1:di-DELAY,:])/ \
    (self.ClosePrice[di-DELAY-(d+d_r)-1:di-DELAY,:]*self.adjfactor[di-DELAY-(d+d_r)-1:di-DELAY,:])
    
    #rise：过去d天signedpower(((return<0)?stddev(return,20):close),2)，dXN数组
    #将过去d天的stddevv(return,20)存入dXN数组，然后使用where语句完成判断
    
    return_stddev = np.full((d,alpha01_return.shape[1]),np.nan)
    
    for ii in range(d):
        return_stddev[ii,:] = np.nanstd(alpha01_closeadj[i:d_r-1+ii+1,:],axis = 0,keepdims = 1)
    
    rise = np.full((d,alpha01_return.shape[1]),np.nan)
    for ii in range (d):
        # ---------------------------
        # 改1：
        # rise[ii,:] = np.where(alpha01_return[d_r+ii,:]<0, return_stddev[ii,:], alpha01_closeadj[ii,:])    
        modified_alpha01_closeadj = alpha01_closeadj / np.nanmean(alpha01_closeadj[:-1,:], axis=0, keepdims=True)
        rise[ii,:] = np.where(alpha01_return[d_r+ii,:]<0, return_stddev[ii,:], modified_alpha01_closeadj[ii,:])
    
    alpha01 =self._rank(self._ts_argmax(rise).reshape((1,rise.shape[1])))
    alpha = alpha01[0,] * self.Universe_one.iloc[i,:]
    

    return alpha




def modified_alpha2():
    # ################################################################################
    # alpha2 : (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    # 逻辑含义： -1 * corr(volume的增长率，return的增长率）
    # key words: 量价走势
    # 卖出策略 卖出 交易量 增长率 和 收益率增长率 在过去6 个交易日内的 相关性
    # 
    
    # 回测结果：2015年与市场反向持续跌；2016年相对平稳，2017年走高 
    # 
    ##cum_return_rate         -0.285018
    ##final_return_rate      -0.0671655
    ##beta                    -0.474139
    ##alpha                  -0.0711064
    ##sharpe ratio            -0.524066
    ##information ratio       -0.428635
    ##turnover rate            0.190379
    ##max drawdown             0.505588
        ##drawdown start date  2015-11-25
        ##drawdown end date          None
    ##fitness                 -0.311279    
    #
    # 改1：volume 本身收到股本规模的影响，此处改为turnover - 换手率（流通股本）
    # ################################################################################ 
    DELAY = self.DELAY 
    d1 = 3
    d2 = 6
    
    # 改1：
    # alpha2_volume = self.Volume[di-DELAY-(d1+d2-1)+1:di-DELAY+1:,:]
    modified_alpha2_volume = self.Turnover[di-DELAY-(d1+d2-1)+1:di-DELAY+1:,:]
    alpha2_closeadj = self.Closeprice[di-DELAY-(d2)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2)+1:di-DELAY+1,:]
    alpha2_openadj = self.Openprice[di-DELAY-(d2)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2)+1:di-DELAY+1,:]

    alpha2 = -1 * self._correlation(self._rank(np.diff(np.log(modified_alpha2_volume), n=2, axis=0)), self._rank((alpha2_closeadj-alpha2_openadj)/alpha2_openadj))

    alpha = alpha2[0,:] * self.Universe_one.iloc[i,:]
    return alpha 




def modified_alpha3():
    
    # ################################################################################
    #alpha3=(-1* correlation(rank(open), rank(volume), 10)) 
    
    #含义 卖出策略
    # 关键词: 量价走势 
    #重点卖出 open和volume在过去10天内相关性较大的股票
    
    # 测试后应该买卖反向——买入 open和volume 相关性较高的股票
    # 
    # 回测结果 （测试为买卖反向后的结果）
    # 2015年以前上涨 幅度超大盘； 2015年 2016年与大盘走势基本一致；2017年后走低
    # cum_return_rate          0.107058
    # final_return_rate       0.0213009
    # beta                     0.450899
    # alpha                  -0.0441551
    # sharpe ratio           -0.0763798
    # information ratio       -0.401445
    # turnover rate            0.149865
    # max drawdown             0.400284
    #   drawdown start date  2017-12-25
    #   drawdown end date          None
    # fitness                -0.0287957
    
    # 改1：volume 本身收到股本规模的影响，此处改为turnover - 换手率（流通股本）
    # ################################################################################
    DELAY = self.DELAY
    
    d = 10
    alpha3_OpenPrice = self.OpenPrice[di-DELAY - d + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha3_RankOpenPrice = self._rank(alpha3_OpenPrice)
    # 改1
    # alpha3_Volume = self.Volume[di-DELAY - d + 1:di-DELAY + 1 , :]
    modified_alpha3_Volume = self.Turnover[di-DELAY - d + 1:di-DELAY + 1 , :]
    
    
    alpha3_RankVolume = self._rank(modified_alpha3_Volume)  #
    alpha3_corr = self._correlation(alpha3_RankOpenPrice , alpha3_RankVolume)
    alpha3 = -1 * alpha3_corr[-1 , :]
    
    # 买卖反向
    alpha3 = -1 * alpha3
    alpha = alpha3 * self.Universe_one.iloc[i , :]

    return alpha 

def modified_alpha4():
    # ################################################################################
    # alpha4 = (-1 * TS_Rank(rank(low),9))
    # 含义： 买入 收盘价 cross大 -- rank值小 -- 相对历史排名 更高 -- ts_rank更小， -1 -->  卖出的权重越小
    #                       小 ---      大 --             低  --         大      -->  卖出的权重越大
    # ==> 卖出策略：重点卖出处于截面和历史低位的股票  
    # 在2周频率上 属于 momentum 行情， 
    # 关键词 Momentumn/Reversal
    
    # 回测结果
    # 持续上涨，除了2015年 有大幅回撤 
    # cum_return_rate          0.754721
    # final_return_rate        0.123593
    # beta                    0.0519674
    # alpha                   0.0846658
    # sharpe ratio              0.93691
    # information ratio       0.0862486
    # turnover rate            0.238967
    # max drawdown             0.166898
    #   drawdown start date  2015-07-08
    #   drawdown end date    2015-09-14
    # fitness                  0.673791
    # 
    # 改1: 这里直接用 low 在截面上比较会受到 上市早晚的影响 用过去8天的 平均low做标准化 
    # 改2：因为这里是现在截面上 rank，然后在 时间上做rank，不需要复权
    # ################################################################################    
    
    DELAY = self.DELAY
    d = 9
    
    # 改2
    # alpha4_low = self.LowPrice[di-DELAY - d + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1:di-DELAY + 1 ,:]
    # alpha4_low = self.LowPrice[di-DELAY - d + 1:di-DELAY + 1 , :]
    
    # 改1
    # alpha4 = -1 * self._ts_rank(self._rank(alpha4_low))
    alpha4 = -1 * self._ts_rank(self._rank(alpha4_low / np.nanmean(alpha4_low, axis=0, keepdims=True)))
    alpha4 = alpha4.reshape(alpha4_low.shape[1] , )
    
    alpha = alpha4 * self.Universe_one.iloc[i , :]

    return alpha 

def modified_alpha5():
    # ################################################################################
    # alpha5=(rank(open-(sum(vwap, 10)/10)))*(-1*abs(rank((close-vwap))))
    #
    # 测试后应该买卖反向操作
    #  买入当天收跌且开盘低于过去10天均价的股票 -- 超跌反弹 
    
    # 关键词 Momentum/Reversal 
    # 在1天的频率下 为 reversal
    
    # 回测结果
    # 持续上涨 -- 关注下为什么
    # cum_return_rate           2.00893
    # final_return_rate        0.256448
    # beta                    0.0437328
    # alpha                    0.218069
    # sharpe ratio              2.28373
    # information ratio        0.608949
    # turnover rate            0.372782
    # max drawdown            0.0812375
    #   drawdown start date  2015-07-08
    #   drawdown end date    2015-07-14
    # fitness                   1.89416
    
    # 改1：这里价差指标可能收到 上市早晚的影响，用过去10天的vwap 作为调整 
    # 改2：降序排列改为升序排列，对应反向暂时取消
    # 改3：abs()函数无用，去掉
    # ################################################################################  
    DELAY = self.DELAY
    d = 10
    alpha5_open = self.openprice[di-DELAY:di-DELAY + 1 , :] * self.adjfactor[di-DELAY:di-DELAY + 1 , :]
    alpha5_vwap = self.vwap[di-DELAY - d + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha5_vwap_last = self.vwap[di-DELAY:di-DELAY + 1 , :]
    alpha5_close = self.closeprice[di-DELAY:di-DELAY + 1 , :]
    
    # 改1|2|3
    # alpha5 = (self._rank(alpha5_open - np.nanmean(alpha5_vwap , axis=0 , keepdims=True))) * (-1) * (np.abs(self._rank(alpha5_close - alpha5_vwap_last)))
    
    alpha5_mean_vwap = np.nanmean(alpha5_vwap, axis=0, keepdims=True)
    alpha5 = (self._rank((alpha5_open - alpha5_mean_vwap)/alpha5_mean_vwap),True) * (-1) * self._rank((alpha5_close - alpha5_vwap_last)/alpha5_mean_vwap,True)
    # 改3
    # alpha5 = alpha5 * (-1)
    alpha = alpha5[0,:] * self.Universe_one.iloc[i , :]

    return alpha 


def modified_alpha6():
    # ################################################################################
    # alpha6 = (-1*correlation(open, volume, 10)
    # 含义 买入过去10天内开盘价格与成交量的相关系数较小的股票
    # 12周的考察期
    # 关键词 量价走势 
    # 测试后应该买卖反向——买入 open和volume 相关性较低的股票
    
    # 回测结果 （最近一年的表现较差）
    # 2015年以前持续温和上涨； 2015年暴涨暴跌； 2016 高位震荡 ； 2017 持续走低
    # cum_return_rate          0.800496
    # final_return_rate        0.129605
    # beta                     0.693569
    # alpha                   0.0480117
    # sharpe ratio             0.339282
    # information ratio        0.119227
    # turnover rate            0.137946
    # max drawdown             0.435465
    #   drawdown start date  2017-12-05
    #   drawdown end date          None
    # fitness                  0.328864
    
    # 改1：将 open用过去9天的均值做调整，
    # 改2：volume 改为 turnover
    # ################################################################################    
    DELAY = self.DELAY
    d = 10
    alpha6_OpenPrice = self.OpenPrice[di-DELAY - d + 1: di-DELAY + 1 , :] * self.adjfactor[ di-DELAY - d + 1: di-DELAY + 1 , :]
    # 改2
    # alpha6_Volume = self.Volume[di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha6_Volume = self.Turnover[di-DELAY - d + 1: di-DELAY + 1 , :]
    
    #改1
    # alpha6 = (-1) * self._correlation(alpha6_OpenPrice , alpha6_Volume)
    alpha6 = (-1) * self._correlation(alpha6_OpenPrice/np.nanmean(alpha6_OpenPrice[:-1,:], axis=0, keepdims=True) , alpha6_Volume)
    
    # 买卖反向
    alpha6 = alpha6 * -1
    
    alpha6 = alpha6[-1 , :]
    alpha = alpha6 * self.Universe_one.iloc[i , :]

    return alpha 


def modified_alpha7():
    # ################################################################################
    # Alpha 7
    # alpha7=((adv20 < volume)? ((-1 * ts_rank(abs(delta(close,7)), 60))*sign(delta(close, 7))): -1*1)
    # 含义： 买卖策略 
    #  交易量 < 过去20天均值 -->  收盘价相对于7天前的变动在过去60天内上升/下降值大小* -1 ==> 买入下降的；卖出上升的 
    #  交易量 >= 过去20天均值 --> Alpha = -1
    
    # 测试后 应买卖方向
    # 如果交易量大于过去20天的均值--> 买入（等权）--牛市下逻辑比价正确；大概率上涨
    # 如果交易量较低：--> 收盘价相对7天前上升--买入； 收盘价相对7天前下降--卖出， 价格变化的幅度越小，买入/卖出的比例越大
    # 关键词： 量价走势; Momentum/Reversal 
    # 在7天的频率下 符合  Momentum --> 如果交易量下降 则确认了momentum的存在 
    
    # 回测结果
    # cum_return_rate           1.36504
    # final_return_rate         0.19529
    # beta                       0.7517
    # alpha                    0.109831
    # sharpe ratio              0.55634
    # information ratio        0.408191
    # turnover rate            0.161276
    # max drawdown             0.401758
    #   drawdown start date  2015-09-15
    #   drawdown end date          None
    # fitness                  0.612202
    
    # 
    # 改1：volume 改为 turnover
    # 改2：delta(close, 7)用过去7天close做标准化
    # ################################################################################  
    
    DELAY = self.DELAY 
    # 首先提取 过去67天的数据
    d1 = 7  # 相对于7天前， index 差 7+1
    d2 = 60
    d3 = 20
    alpha7_ClosePrice = self.ClosePrice[di-DELAY - (d1 + 1) - d2 + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - ( d1 + 1) - d2 + 1:di-DELAY + 1 , :]
    
    # 计算60个横截面组合，然后放入 _ts_rank
    delta_closprice = np.zeros((d2 , alpha7_ClosePrice.shape[1]))
    delta_closprice.fill(np.nan)
    for start in range(d2):
        end = start + d1 + 1
        # 改2
        delta_closprice[start] = self._delta(alpha7_ClosePrice[start:end])/np.nanmean(alpha7_ClosePrice[start:end-1])
    
    alpha7_tsrank = self._ts_rank((np.abs(delta_closprice)))
    # 改1
    # alpha7_volume = self.Volume[di-DELAY - d3:di-DELAY , :]
    alpha7_volume = self.Turnover[di-DELAY - d3:di-DELAY , :]
    alpha7_condition = np.nanmean(alpha7_volume , axis=0 , keepdims=True) - self.Volume[di-DELAY:di-DELAY + 1 , :]

    alpha7 = np.where(alpha7_condition < 0 , (-1) * alpha7_tsrank * np.sign(delta_closprice[-1 , :].reshape(1 ,-1)) , -1)
    
    # 买卖反向
    alpha7 = alpha7 * -1
    alpha = alpha7[0,:] * self.Universe_one.iloc[i , :]

    return alpha 



def modified_alpha8():
    
    # ################################################################################
    # alpha8=(-1 * rank(((sum(open, 5) * sum(returns 5)) - delay((sum(open,5) * sum(returns,5)), 10))))
    # 基于前15天的数据计算 先计算 相对于过去5天的一个指标，这个指标减去15天前的这个指标 -- > 在cross-section 上做排序
    # 
    # 
    # 含义 卖出策略
    # 卖出：open * return 可以看成是 delta price的一个变形 卖出 delta price 相对于10天前 下降明显的股票  
    #       1.大量卖出大幅下跌（相对）的 
    #       2.少量卖出大幅上涨（相对）的
    # 关键词  Momentum/Reversal --> 认为在10天的频率下 存在 reversal  ==> 可能需要反向  前面验证的逻辑是momentum？
    #
    # 回测结果 买卖反向后 
    # 2015年以前稳定上升；2015暴涨暴跌；2016震荡上行 2017 震荡下行 
    ##cum_return_rate           1.36878
    ##final_return_rate        0.195681
    ##beta                     0.906451
    ##alpha                   0.0999307
    ##sharpe ratio             0.450629
    ##information ratio        0.336602
    ##turnover rate            0.107431
    ##max drawdown             0.534904
        ##drawdown start date  2017-12-05
        ##drawdown end date          None
    ##fitness                  0.608174
    # ################################################################################
    
    d1 = 5 
    d2 = 10 
    alpha8_openprice = self.Openprice[di-DELAY-d1-(d2+1)+1:di-DELAY]*self.adjfactor[di-DELAY-d1-(d2+1)+1:di-DELAY]
    alpha8_return = self.Pctchange[di-DELAY-d1-(d2+1)+1:di-DELAY]
    
    # 计算 (sum(open, 5) - sum(returns 5)
    open_return = (self._ts_sum(alpha8_openprice[-5:,:]) * self._ts_sum(alpha8_return[-5:,:]))  
    open_return10 = (self._ts_sum(alpha8_openprice[-15:-10,:]) * self._ts_sum(alpha8_return[-15:-10,:]))
    alpha8 = -1 * self._rank(open_return - open_return10)
    # 买卖反向 
    alpha8 = alpha8 * -1 
    
    alpha = alpha8[0,:]*self.Universe_one.iloc[i,:]







def modified_alpha9():
    
    # ################################################################################
    # Alpha 9 = ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    #
    # 含义： 过去5天内 收盘价变动的最小值 > 0 --> 买入当日收盘价大幅上升的股票；卖出当日收盘价大幅下跌的股票
    #                                <=0 --> 过去5天内收盘价变动 的最大值 < 0 --> 卖出当日收盘价大幅下跌的股票
    #                                                                 >=0 --> 卖出当日收盘价大幅上升的股票；买入当日收盘价大幅下跌的股票
    # 关键词  Momentum/Reversal: 以1天为考察频率，如果过去5天内 收盘价表现为全部上涨 --> 买入 变动幅度越大 买入权重越大 (Momentum) 
    #                                         如果过去5天内               下跌 --> 卖出 并且当天跌幅越大 卖出权重越大 (Momentum)
    #                                         如果过去5天有涨有跌               --> 更符合 Mean-reverting （reversal)
    # 回测结果 2015年以前 稳定上涨；2015年波动较大 但上涨幅度小 暴跌程度大； 2015下半年以后持续上涨至2016年底； 2017年继续下跌，2017年10月以后开始稳定回升
    #         有大量股票超配 > 5%
    ##cum_return_rate           1.06151
    ##final_return_rate        0.161744
    ##beta                     0.056799
    ##alpha                    0.122496
    ##sharpe ratio             0.384289
    ##information ratio        0.148789
    ##turnover rate            0.526537
    ##max drawdown             0.448677
      ##drawdown start date  2015-08-27
      ##drawdown end date    2016-06-28
    ##fitness                  0.212989
    # ################################################################################
    
    d0 = 2
    d1 = 5 
    
    alpha9_closeadj = self.Closeprice[di-DELAY-(d0+d1-1)+1:di-DELAY+1,:]*self.adjfactor[di-DELAY-(d0+d1-1)+1:di-DELAY+1,:]
    alpha9_closeadj_delta = np.diff(alpha9_closeadj,axis = 0)
    
    alpha9_cond1 = 0 < self._ts_min(alpha9_closeadj_delta)
    alpha9_cond2 = 0 >  self._ts_max(alpha9_closeadj_delta)
    
    alpha9_value1 = alpha9_closeadj_delta[-1,:]
    alpha9_value2 = alpha9_value1 
    alpha9_value3 = -1 * alpha9_value1
    
    alpha9 = np.where(alpha9_cond1, alpha9_value1, (np.where(alpha9_cond2, alpha9_value2, alpha9_value3)))
    
    alpha = alpha9[0,:] * self.Universe_one.iloc[i,:]
    return alpha 






def modified_alpha10():
    
    # ################################################################################
    # Alpha 10 = rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
    #
    # 与alpha9 类似，但是是基于过去4天而不是5天的信息做判断 
    # alpha9 效果跟好
    # 回测结果 有大量超配 >5% 2015以前 稳定上涨； 2015年暴涨暴跌 但涨幅不大 ； 2015下半年-2016 持续上涨 2017前半年持续下跌， 2017下半年回升
    ##cum_return_rate          0.526599
    ##final_return_rate       0.0916278
    ##beta                    0.0988396
    ##alpha                   0.0495838
    ##sharpe ratio             0.176089
    ##information ratio      -0.0267015
    ##turnover rate            0.525122
    ##max drawdown             0.568866
      ##drawdown start date  2015-08-27
      ##drawdown end date    2016-10-10
    ##fitness                 0.0735558
    # ################################################################################
    
    d0 = 2
    d1 = 4

    alpha10_closeadj = self.Closeprice[di-DELAY-(d0+d1-1)+1:di-DELAY+1,:]*self.adjfactor[di-DELAY-(d0+d1-1)+1:di-DELAY+1,:]
    alpha10_closeadj_delta = np.diff(alpha10_closeadj,axis = 0)

    alpha10_cond1 = 0 < self._ts_min(alpha10_closeadj_delta)
    alpha10_cond2 = 0 >  self._ts_max(alpha10_closeadj_delta)

    alpha10_value1 = alpha10_closeadj_delta[-1,:]
    alpha10_value2 = alpha10_value1 
    alpha10_value3 = -1 * alpha10_value1

    alpha10 = np.where(alpha10_cond1, alpha10_value1, (np.where(alpha10_cond2, alpha10_value2, alpha10_value3)))

    alpha = alpha10[0,:] * self.Universe_one.iloc[i,:]
    return alpha 









def modified_alpha11():
    
    # ################################################################################
    # Alpha 11 = ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    #
    # 含义：买入策略 
    # 关键字: Momentum/Reversal 
    # 3日内的趋势 ，(1)（当日均价-收盘价）在过去三天的最大值 比较负（负值 绝对值大） 且 (2)  当日均价-收盘价）在过去三天的最小值 比较小  (3) volume 相对于前3天有大幅下降 
    # 大量买入 过去3天内尾盘下跌的股票 Reversal -- 交易量下降 确认 Reversal 
    # 这个似乎和前面某个策略矛盾？
    # 
    # 回测结果 2015年以前大幅上涨；2015暴涨暴跌 2016震荡上升 2017震荡下降 
    ##cum_return_rate           1.22045
    ##final_return_rate        0.179765
    ##beta                     0.881359
    ##alpha                    0.085683
    ##sharpe ratio             0.420327
    ##information ratio        0.289967
    ##turnover rate            0.222233
    ##max drawdown             0.546254
        ##drawdown start date  2015-09-15
        ##drawdown end date          None
    ##fitness                  0.378038    
    # ################################################################################
    
    d = 3 

    alpha11_vwapadj = self.vwap[di-DELAY-d+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d+1:di-DELAY+1,:] 
    alpha11_closeadj = self.Closeprice[di-DELAY-d+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d+1:di-DELAY+1,:] 
    alpha11_volume = self.Volume[di-DELAY-d+1:di-DELAY+1,:] 

    alpha11 = (self._rank(self._ts_max(alpha11_vwapadj - alpha11_closeadj)) + self._rank(self._ts_min(alpha11_vwapadj - alpha11_closeadj))) * self._rank(self._delta(alpha11_volume))

    alpha = alpha11[0,:] * self.Universe_one.iloc[i,:]
    return alpha 





def modified_alpha12():
    
    # ################################################################################
    # Alpha 12
    # alpha12=(sign(delta(volume,1) * (-1 * delta(close,1)))
    # 含义: 买卖策略
    # 买入：volume 变动方向与close变动方向相反 量增 + 价跌 or 量减 + 价升
    # 卖出：volume 变动方向与close变动方向相同 量增 + 价升 or 量减 + 价跌
    # 关键词：Momentum/Reversal； 量价关系
    
    # 可以从 量价相关性 也可以从 Momentum/Reversal的角度理解；Volume用于判断M/哪个成立     
    # a) 第一种理解： 买入 volume 和价格的变动方向相反的股票 -- 理论上也是相关性程度比较低
    # b) 第二种理解： 1天周期，量升 --> reversal； 量降 --> Momentum 
    #
    #
    #
    # 测试结果 2015年以前上升；2015行情大涨时暴跌；暴跌时持平震荡；2016震荡后暴跌；2017持续上升 与市场走势基本一致
    # cum_return_rate          0.244207
    # final_return_rate       0.0463217
    # beta                    -0.316772
    # alpha                   0.0319159
    # sharpe ratio            0.0382272
    # information ratio       -0.131545
    # turnover rate            0.494245
    # max drawdown             0.442178
    #   drawdown start date  2015-05-27
    #   drawdown end date          None
    # fitness                 0.0117029
    # ################################################################################
    
    # 首先提取 过去67天的数据
    d1 = 7  # 相对于7天前， index 差 7+1
    d2 = 60
    d3 = 20
    alpha7_ClosePrice = self.ClosePrice[di-DELAY - (d1 + 1) - d2 + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - (d1 + 1) - d2 + 1:di-DELAY + 1 , :]
    
    alpha12_volume = self.Volume[di-DELAY - 1:di-DELAY + 1 , :]
    alpha12_close = self.Closeprice[di-DELAY - 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - 1:di-DELAY + 1 , :]
    alpha12 = -1 * np.sign(self._delta(alpha12_volume)) * self._delta(alpha12_close)
    alpha = alpha12.reshape(alpha12_volume.shape[1] , ) * self.Universe_one.iloc[i , :]
    return alpha 






def modified_alpha13():
    # ################################################################################
    # alpha13=(-1*rank(covariance(rank(close), rank(volumne), 5)))
    # 含义: # 卖出策略
    # 卖出：close 和 volume 的相关性小--> rank的数值大 重点卖出
    # 
    #
    # 测试后改为买入策略： 
    #5天考察周期，买入close和volume相关性比较小的股票
    # 回测结果
    # ------------------------------------------------
    # cum_return_rate           1.93167
    # final_return_rate        0.249693
    # beta                      1.01607
    # alpha                    0.146654
    # sharpe ratio             0.542739
    # information ratio        0.482584
    # turnover rate            0.107526
    # max drawdown             0.571679
    #   drawdown start date  2015-09-15
    #   drawdown end date          None
    # fitness                   0.82706
    # ################################################################################

    d = 5
    alpha13_volume = self.Volume[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha13_close = self.Closeprice[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha13 = -1 * self._rank(self._covariance(self._rank(alpha13_close) , self._rank(alpha13_volume)))
    alpha13 = alpha13 * -1
    alpha = alpha13.reshape(alpha13_volume.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha










def modified_alpha14():
    
    # ################################################################################
    # alpha14=(-1* rank(delta(returns, 3)))*correlation(open, volume, 10))
    # 含义:  买卖策略
    # 买入：过去10天量价相关性为负数，买：return大幅度下降+量价背离程度非常大
    # 卖出：过去10天量价相关性为正数--很少发生？，卖：return答复下降+加量趋同程度非常大
    
    # 测试后 买卖反向
    #  --卖出 过去10天 open 和 volume 相关性为负 -->  delta(return)越负： return 相比于3天前大幅下降+ 价格和量的负相关越大， 卖出越多；
    #  --买入 过去10天 open 和 volume 相关性为正 -->  delta(return)越负： return 相比于3天前大幅下降+ 价格和量的正相关程度越大 买入越多
    # 关键词: Momentum/Reversal + 量价关系 
    # 
    # 主要买卖过去相比于3天前的return大幅下降的股票，如果在过去10天内量价同步 则大量买入 --确认 reversal 
    #                                                          异步 则大量卖出  --确认 momentum
    
    # 测试结果
    # cum_return_rate           1.15866
    # final_return_rate        0.172884
    # beta                     0.693656
    # alpha                   0.0912848
    # sharpe ratio             0.491166
    # information ratio        0.302369
    # turnover rate            0.317091
    # max drawdown             0.376082
    #   drawdown start date  2017-12-05
    #   drawdown end date          None
    # fitness                  0.362672
    # ################################################################################    
    d1 = 3
    d2 = 10
    alpha14_open = self.Openprice[di-DELAY - d2 + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    alpha14_volume = self.Volume[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    alpha14_close = self.Closeprice[di-DELAY - d1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d1:di-DELAY + 1 , :]
    alpha14_return = np.diff(alpha14_close , axis=0) / alpha14_close[:-1]
    
    alpha14 = -1 * self._rank(self._delta(alpha14_return)) * self._correlation(alpha14_open , alpha14_volume)
    
    # 买卖操作反向
    alpha14 = alpha14 * -1
    alpha = alpha14.reshape(alpha14_volume.shape[1] , ) * self.Universe_one.iloc[i , :]
    
    return alpha 






def modified_alpha15():
    # ################################################################################
    # alpha15= (-1* sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    # 含义: # 卖出策略
    # 卖出：high和volume 相关程度越小 -- > rank 的数值越大，卖出的比例越大
    # 关键词：量价相关性 
    # 测试后 应买卖反向操作：买入 high 和 volume相关性程度较低的股票
    # 
    # 本质上还是看的是过去5天内的correlation，但是是按照3天计算一个correlation 累加 
    # 回测结果 2015年以前上涨；2015暴涨暴跌，2016|2017 震荡下行
    # cum_return_rate            1.5646
    # final_return_rate        0.215525
    # beta                      1.03224
    # alpha                     0.11141
    # sharpe ratio             0.447618
    # information ratio        0.363169
    # turnover rate           0.0839657
    # max drawdown             0.577177
    #   drawdown start date  2015-09-01
    #   drawdown end date          None
    # fitness                  0.717142
    # ################################################################################    
    d = 5
    alpha15_high = self.Highprice[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha15_volume = self.Volume[di-DELAY - d + 1:di-DELAY + 1 , :]
    
    alpha15_corr1 = self._correlation(self._rank(alpha15_high[0:3 , :]) , self._rank(alpha15_volume[0:3 , :]))
    alpha15_corr2 = self._correlation(self._rank(alpha15_high[1:4 , :]) , self._rank(alpha15_volume[1:4 , :]))
    alpha15_corr3 = self._correlation(self._rank(alpha15_high[2:5 , :]) , self._rank(alpha15_volume[2:5 , :]))
    alpha15_corr = np.row_stack((alpha15_corr1 , alpha15_corr2 , alpha15_corr3))
    alpha15 = -1 * self._ts_sum(self._rank(alpha15_corr))
    
    # 买卖反向操作
    alpha15 = -1 * alpha15
    alpha = alpha15.reshape(alpha15_volume.shape[1] , ) * self.Universe_one.iloc[i , :]
    
    return alpha







def modified_alpha16():
    # ################################################################################
    # alpha16=(-1*rank(covariance(rank(close), rank(volumne), 5)))
    # 卖出策略
    # 卖出：close 和 volume 的相关性小--> rank的数值大 重点卖出
    #
    # 关键词：量价相关性 
    # 测试之后，应该买卖反向操作-- 买入 close 和 volume相关性较低的股票
    # 
    # 回测结果: (买卖反向之后的结果) 2015年以前上涨；2015暴涨暴跌，2016|2017 震荡下行
    # ------------------------------------------------
    # cum_return_rate           1.91327
    # final_return_rate        0.248064
    # beta                      1.02547
    # alpha                    0.144398
    # sharpe ratio             0.532125
    # information ratio        0.470559
    # turnover rate            0.105902
    # max drawdown             0.572779
    #   drawdown start date  2015-09-15
    #   drawdown end date          None
    # fitness                  0.814412
    # ################################################################################    
    d = 5
    alpha16_volume = self.Volume[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha16_high = self.Highprice[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha16 = -1 * self._rank(self._covariance(self._rank(alpha16_high) , self._rank(alpha16_volume)))
    
    # 买卖反向
    alpha16 = alpha16 * -1
    alpha = alpha16.reshape(alpha16_volume.shape[1] , ) * self.Universe_one.iloc[i , :]
    
    return alpha 




def modified_alpha17():

       
    # ################################################################################
    # alpha17=(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    #
    # 含义(反向后)：买入策略 
    # close位于截面和过去10天的高位 * 昨天的收盘价 相比于 今天和前天的均价的差值更大 （1.今天收盘价下跌；2.昨天收盘价相比于前天收盘价大涨） * 交易量相比基准（20天均值）在过去5天内处于高位
    # 逻辑：昨天涨，今天跌 -- 1天内的反转 （现在的价格处于过去10天的高位+今天的交易量处于过去5天的高位 --> 辅助确认 Reversal )
    # Momentum/Reversal 
    # 回测结果 （反向之后） (买卖反向之后的结果) 2015年以前上涨；2015暴涨暴跌，2016|2017 震荡下行
    #
    # 201
    ##cum_return_rate           1.55951
    ##final_return_rate        0.215025
    ##beta                     0.970911
    ##alpha                    0.114988
    ##sharpe ratio             0.473585
    ##information ratio        0.383088
    ##turnover rate            0.314061
    ##max drawdown             0.547133
      ##drawdown start date  2015-09-15
      ##drawdown end date          None
    ##fitness                  0.391864
    # ################################################################################    
    
    d1 = 10 
    d2 = 2 + 1
    d0 = 20 
    d3 = 5 

    alpha17_closeadj = self.Closeprice[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:] 
    alpha17_volume = self.Volume[di-DELAY-(d0+d3-1)+1:di-DELAY+1,:]

    alpha17_rank1 = self._rank(self._ts_rank(alpha17_closeadj))
    alpha17_rank2 = self._rank(np.diff(np.diff(alpha17_closeadj[-3:,:], axis=0), axis=0))

    for ii in range(d3):
        jj = ii + d0 
        alpha17_adv20 = np.nanmean(alpha17_volume, axis=0, keepdims=True)

    alpha17_rank3 = self._rank(self._ts_rank(alpha17_adv20))

    alpha17 = -1 * alpha17_rank1 * alpha17_rank2 *alpha17_rank3
    # 测试买卖反向 
    alpha17 = alpha17 * -1
    alpha = alpha17[0,:] * self.Universe_one.iloc[i,:]
    return alpha 





def modified_alpha18():
    
    
    # ################################################################################
    # alpha18=(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
    # 含义（反向后）：买入策略
    # 买入：收盘相对开开盘大幅下降 + 收开价差的波动率较小 + 收开价格的相关系数较小 
    # 关键词：Momentum/Reversal 
    # 逻辑：当过去五天 收盘|开盘价差绝对值的波动较小 （相对平稳，变化不大）+ 过去10日量价的相关性较低 --> （辅助确认下跌后的reversal) 突然出现大幅的日内下跌 
    # 本质是 1天的reversal 策略，需要量价关系和价格变动波动性 来辅助判断 reversal 发生的概率是否足够大 
    # 
    # 1. 直接相加 似乎不太好，因为 correlation显然和另外两个不在一个量纲上，主要依赖 收开价差波动性辅助判断Reversal; 
    #     但从前面的结果来看，量价相关性本身似乎很有效，但量价相关性 之前都是直接作为买入信号的，越小 越值得买入 这个是一致的 
    #    可以考虑将三个指标分别求rank之后再相乘或者相加  
    #
    # 测试之后，应该买卖反向操作--
    #
    # 回测结果: (买卖反向之后的结果)
    # ------------------------------------------------
    #
    # final_return_rate        0.330907
    # beta                     0.964886
    # alpha                    0.231271
    # sharpe ratio             0.795644
    # information ratio        0.799734
    # turnover rate            0.211636
    # max drawdown             0.519344
    #   drawdown start date  2015-09-15
    #   drawdown end date          None
    # fitness                  0.994893
    # ################################################################################
    
    d1 = 5
    d2 = 10
    alpha18_open = self.Openprice[di-DELAY - d2 + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    alpha18_close = self.Closeprice[di-DELAY - d2 + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    alpha18_1 = (np.nanstd(np.abs(alpha18_close[-d1:] - alpha18_open[-d1:]) , axis=0)).reshape(1 , alpha18_open.shape[1])
    alpha18_2 = alpha18_close[d2 - 1:d2 , :] - alpha18_open[d2 - 1:d2 , :]
    alpha18_3 = self._correlation(alpha18_close , alpha18_open)
    alpha18 = -1 * self._rank(alpha18_1 + alpha18_2 + alpha18_3)
    
    # 买卖操作反向
    alpha18 = alpha18 * -1
    alpha = alpha18.reshape(alpha18_open.shape[1] , ) * self.Universe_one.iloc[i , :]
    return alpha 








def modified_alpha19():
    
    # ################################################################################
    # alpha19 = ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    #
    # 含义（反向后）
    #
    # 当 (close - 7天前 close + close 减去7天前的close ) > 0 买入  --> 250天的return较小的股票 权重较大
    # 当 (close - 7天前 close + close 减去7天前的close ) < 0 卖出  --> 250天的return较小的股票 权重较大
    #    return 越正 --> 价格持续升高； return越负 -- > 价格持续下降
    
    # 买入：相对七天前上涨，且过去一年整体下跌 
    # 卖出：相对七天前下跌，且过去一年整体上涨    
    
    # 关键字：Momentum/Reversal； 不同周期价格走势 7天内的Momentum （长期中处于单边行情时满足）
    
    # 1周的走势 和1年的走势发生了背离 
    # 逻辑：1天频率下保持momentum（相对于7天之前的变动），return较负 表明在持续下行 辅助确认momentum的大小 
    # 回测结果 (买卖反向) -- 15年暴涨暴跌期间不错
    #
    
    # cum_return_rate          0.688937
    # final_return_rate        0.114731
    # beta                   -0.0754456
    # alpha                   0.0842766
    # sharpe ratio             0.287533
    # information ratio       0.0334702
    # turnover rate            0.150435
    # max drawdown             0.313036
    #   drawdown start date  2015-02-25
    #   drawdown end date    2015-06-02
    # fitness                  0.251104
    # ################################################################################
    
    d1 = 7  # 7天以前，共计取8天的数据
    d2 = 250  # 共计取251天的数据
    alpha19_close = self.Closeprice[di-DELAY - d1:di-DELAY + 1 , :]
    alpha19_adjclose = self.Closeprice[di-DELAY-d2:di-DELAY+1 , :] * self.adjfactor[di-DELAY - d2:di-DELAY+1,:]
    alpha19_return = np.diff(alpha19_adjclose , axis=0) / alpha19_adjclose[:-1]
    alpha19 = -1 * np.sign(self._delta(alpha19_close)) * self._rank(1 + np.nanmean(alpha19_return , axis=0 , keepdims=True))
    alpha = alpha19.reshape(alpha19_close.shape[1] , ) * self.Universe_one.iloc[i , :]
    
    return alpha 




def modified_alpha20():
    
    # ################################################################################
    # Alpha20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
    # 含义（反向后）买卖策略
    # 买入/卖出 开盘价格高于前一天的价格的股票 （强势）
    # 1天周期 momentum策略； 重仓买入 今日高开的股票 
    # 关键词： Momentum/Reversal 
    # 测试后需要买卖反向
    # 回测结果 2015年温和上涨；2015暴涨暴跌； 2016 震荡  2017 下行 
    #
    # cum_return_rate           0.596718
    # final_return_rate         0.101835
    # beta                      0.975215
    # alpha                   0.00151154
    # sharpe ratio              0.173593
    # information ratio     -0.000458083
    # turnover rate             0.456842
    # max drawdown              0.659061
    #   drawdown start date   2017-12-05
    #   drawdown end date           None
    # fitness                  0.0819591
    #
    # ################################################################################    
    d = 2
    alpha20_close = self.Closeprice[di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha20_open = self.Openprice[di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha20_high = self.Highprice[di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha20_low = self.Lowprice[di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha20 = -1 * self._rank(alpha20_open[1:2 , :] - alpha20_high[0:1 , :]) * self._rank(alpha20_open[1:2 , :] - alpha20_close[0:1 , :]) * self._rank(alpha20_open[1:2 , :] - alpha20_low[0:1 , :])
    
    alpha20 = alpha20 * -1
    alpha = alpha20.reshape(alpha20_close.shape[1] , ) * self.Universe_one.iloc[i , :]
    return alpha 


def modified_alpha21():
    
    # ################################################################################
    # Alpha21: ((((sum(close , 8) / 8) + stddev(close , 8)) < (sum(close , 2) / 2)) ? (-1 * 1): (((sum(close , 2) / 2) < ((sum(close , 8) / 8) - stddev(close , 8))) ? 1: (((1 < (volume / adv20)) | | ((volume / adv20) == 1)) ? 1: (-1 * 1))))
    # 含义 买卖策略
    # 过去2日的均价>过去8日的均价+1个标准差 （短期价格处于高位） --> 等权卖出 
    # 过去2日的均价<过去8日的均价-1个标准差 （短期价格处于低位） --> 等权买入        
    # 过去2日的均价=过去8日的均价+1个标准差 （短期价格处于中等） --> 当日交易量 >= 过去20天均值 --> 等权买入 
    #                                                                 <             --> 等权卖出
    # 关键词：Momentum/Reversal 
    # 逻辑：本质还是7天左右的Reversal策略 认为没有其他辅助的情况下 相对7天前指标计算的信号是满足reversal的
    #
    # 注意与alpha19 对比， alpha19中是 momentum，但momentum成立有一个必要的条件就是长期内的价格处于相反的单边行情中
    #回测结果 
    
    ##cum_return_rate          0.688146
    ##final_return_rate        0.114622
    ##beta                     0.884698
    ##alpha                   0.0203186
    ##sharpe ratio             0.229757
    ##information ratio       0.0470149
    ##turnover rate          0.00499012
    ##max drawdown             0.568185
        ##drawdown start date  2017-12-05
        ##drawdown end date          None
    ##fitness                   1.10115
    # ################################################################################
    
    d1 = 8 
    d2 = 20

    alpha21_closeadj = self.Closeprice[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:]
    alpha21_volume = self.Volume[di-DELAY-d2+1:di-DELAY+1,:]

    alpha21_cond1 = np.nanmean(alpha21_closeadj, axis=0, keepdims=False) + np.nanstd(alpha21_closeadj) < np.nanmean(alpha21_closeadj[-2:,:], axis=0, keepdims=False)
    alpha21_cond2 = np.nanmean(alpha21_closeadj, axis=0, keepdims=False) - np.nanstd(alpha21_closeadj) > np.nanmean(alpha21_closeadj[-2:,:], axis=0, keepdims=False)

    alpha21_adv20 = np.nanmean(alpha21_volume, axis=0, keepdims=False)
    alpha21_cond3 = alpha21_volume[d2-1:d2,:] >= alpha21_adv20

    alpha21_value1 = -1 * np.ones((alpha21_closeadj.shape[1]))
    alpha21_value2 = np.ones((alpha21_closeadj.shape[1]))
    alpha21_value3 = alpha21_value2
    alpha21_value4 = alpha21_value1

    alpha21 = np.where(alpha21_cond1, alpha21_value1, (np.where(alpha21_cond2, alpha21_value2, np.where(alpha21_cond3, alpha21_value3, alpha21_value4))))

    # 注意 alpha21 已经是 array 格式
    alpha = alpha21[0,:] * self.Universe_one.iloc[i,:]
    return alpha 




def modified_alpha22():
    # ################################################################################
    # Alpha22: (-1 * (delta(correlation(high , volume , 5) , 5) * rank(stddev(close , 20))))
    # 含义 ：买卖策略
    # 买入：本周 量价相关系数 下降的股票 -- 过去20天股价波动幅度越小，权重越大
    # 卖出：本周 量价相关系数 上升的股票 -- 过去20天股价波动幅度越小，权重越大
    
    # 关键词  量价走势的动态变化
    # 回测结果 (原策略收益率为负数-->测试买卖反向之后的结果：在2015暴跌之后完全失效)
    
    ##cum_return_rate          -0.55892
    ##final_return_rate       -0.156022
    ##beta                     0.058256
    ##alpha                   -0.195368
    ##sharpe ratio             -1.04103
    ##information ratio       -0.874291
    ##turnover rate            0.352753
    ##max drawdown             0.734976
        ##drawdown start date  2016-06-13
        ##drawdown end date          None
    ##fitness                 -0.692341
    # ################################################################################    
    d1 = 5
    d2 = 10
    d3 = 20
    
    alpha22_high = self.Highprice[di-DELAY - d2 + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    alpha22_volume = self.volume[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    
    alpha22_close = self.Closprice[di-DELAY - d3 + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d3 + 1:di-DELAY + 1 , :]
    
    corr_st = self._correlation(alpha22_high[1:6 , :] , alpha22_volume[1:6 , :])
    corr_ed = self._correlation(alpha22_high[5:10 , :] , alpha22_volume[5:10 , :])
    
    alpha22 = -1 * (corr_ed - corr_st) * self._rank(np.nanstd(alpha22_close , axis=0).reshape(1 , alpha22_close.shape[1]))
    
    # 买卖反向
    alpha22 = -1 * alpha22    
    alpha = alpha22[0,:] * self.Universe_one.iloc[i , :]
    return alpha






def modified_alpha23():
    # ################################################################################
    # Alpha23: (((sum(high , 20) / 20) < high) ? (-1 * delta(high , 2)): 0)
    
    # 含义 （反向后）买卖策略
    # 当日最高价 > 过去20天最高价的均值 （最高价向上穿过benchmark）--> 买入 超过3天前最高价的股票 # 卖出低于3天前最高价的股票 
    #          <=                                              无操作      
    # 逻辑：最高价大于过去20天的均值--> 确认3天内存在Momentum 
    #  这里似乎逻辑上不太对，如果处于历史的高位 是不是更有可能出现reversal  
    #
    #回测结果（买卖反向）股票超配 -- 2015年以前稳定上涨；2015年暴涨暴跌；2016年震荡；2017年下行
    # 
    #
    # cum_return_rate          0.341497
    # final_return_rate        0.062775
    # beta                     0.797009
    # alpha                  -0.0256973
    # sharpe ratio            0.0818546
    # information ratio       -0.142601
    # turnover rate            0.134024
    # max drawdown             0.700185
    #   drawdown start date  2017-12-27
    #   drawdown end date          None
    # fitness                 0.0560203
    # ################################################################################
    
    d = 20
    alpha23_high = self.Highprice[di-DELAY - d + 1:di-DELAY + 1 , :]
    
    cond = np.nanmean(alpha23_high , axis=0 , keepdims=True) < alpha23_high[d - 1:d , :]
    alpha23 = np.where(cond , (-1 * alpha23_high[d - 1:d , :] - alpha23_high[d - 3 , d - 2]) ,np.zeros(alpha23_high.shape[1])).reshape(1 , alpha23_high.shape[1])
    
    # 买卖反向
    alpha23 = -1 * alpha23
    alpha = alpha23.reshape(alpha23_high.shape[1] , ) * self.Universe_one.iloc[i , :]
    return alpha 



def modified_alpha24():
    # ################################################################################
    # Alpha24: ((((delta((sum(close , 100) / 100) , 100) / delay(close , 100)) < 0.05) | | ((delta((sum(close , 100) / 100) , 100) / delay(close , 100)) == 0.05))
    #         ? (-1 * (close - ts_min(close , 100))): (-1 * delta(close , 3)))
    # 
    # 含义: 100天内收盘均价相比于100天前的变动/100天前的收盘价 <=0.05 --> 买入：收盘价位于过去100天最低       卖出：收盘价高于过去100天最低 （100天内的极端反转）
    #      100天内收盘均价相比于100天前的变动/100天前的收盘价 > 0.05 --> 买入：收盘价相对三天前下跌的     卖出  收盘价相对于三天前上涨的  （3天内的反转）
    # 逻辑：价格温和上涨或者下跌 --> 买入极端最低的股票；卖出其他股票；*** 温和上涨期找100天前的价格作为基准 (reversal)
    #      价格涨幅超过5% --> 买入相对3天前下跌的                 *** 快速上涨期找近3天的价格作为基准（reversal)
    #      
    # 关键词：Reversal 
    # 可能的问题：1. 每一期 买卖的股票可能不够平衡 2. 逻辑上不太容易理解
    
    # 回测结果 2015年之前与市场走势相反；2016年震荡；17年上行 
    #  2015年以前反向操作应该是可以的 
    ##cum_return_rate         -0.574268
    ##final_return_rate       -0.162194
    ##beta                    -0.546514
    ##alpha                   -0.161322
    ##sharpe ratio            -0.564195
    ##information ratio       -0.529595
    ##turnover rate            0.098849
    ##max drawdown             0.782472
        ##drawdown start date  2015-11-25
        ##drawdown end date          None
    ##fitness                 -0.722704
    # ################################################################################
    d1 = 100
    d2 = 100 
    d3 = 3

    alpha24_closeadj = self.Closeprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]

    alpha24_meanclose = np.zeros((d2,alpha24_closeadj.shape[1]))
    for ii in range(d2):
        jj = ii+d1
        alpha24_meanclose[ii] = np.nanmean(alpha24_closeadj[ii:jj], axis=0, keepdims=True)

    alpha24_cond = ((self._delta(alpha24_meanclose))[0,:] / (alpha24_closeadj[-d2,:])) <= 0.05

    alpha24_value1 = -1 * alpha24_closeadj[d1+d2-2:d1+d2-1,:] - self._ts_min(alpha24_closeadj[-100:,:]) 
    alpha24_value2 = -1 * self._delta(alpha24_closeadj[-d3:,:])

    alpha24 = np.where(alpha24_cond, alpha24_value1, alpha24_value2)
    alpha = alpha24[0,:] * self.Universe_one.iloc[i , :]
    return alpha 



def modified_alpha25():
    # ################################################################################
    # Alpha25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    
    # 含义 买入策略  
    # return > 0 且 rank()内数值的绝对值越大，权重越大 
    # return < 0 也买入 但权重相对>0的股票较低，价格越高、交易量越活跃，最高价相比收盘价的差别越大 等都辅助确认reversal 效应越差  
    # 
    # 逻辑
    #  *大幅买入当天上涨的股票 -- 价格越高、交易量越活跃，最高价相比收盘价的差别越大 等都辅助确认momentum 效应越强 
    #  *1天下 Momentum的效应更强，尤其是加上交易量等的辅助判断之后 
    # 关键词：
    
    
    # 回测结果 2015前温和上涨；2015暴涨暴跌；2016震荡；2017 下行 
    # cum_return_rate          0.399064
    # final_return_rate       0.0720695
    # beta                     0.875411
    # alpha                  -0.0216165
    # sharpe ratio             0.107034
    # information ratio       -0.111622
    # turnover rate            0.269237
    # max drawdown             0.588038
    #   drawdown start date  2017-12-05
    #   drawdown end date          None
    # fitness                 0.0553774
    #
    # ################################################################################    
    d = 20
    
    alpha25_volume = self.Volume[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha25_adjclose = self.Closeprice[di-DELAY - 2 + 1: di-DELAY + 1 , :] * self.adjfactor[di-DELAY - 2 + 1: di-DELAY + 1 , :]
    alpha25_close = self.Closeprice[di-DELAY: di-DELAY + 1 , :]
    alpha25_high = self.Highprice[di-DELAY: di-DELAY + 1 , :]
    alpha25_vwap = self.vwap[di-DELAY: di-DELAY + 1 , :]
    alpha25_return = np.diff(alpha25_adjclose , axis=0) / alpha25_adjclose[:-1]
    alpha25 = self._rank(-1 * alpha25_return * np.nanmean(alpha25_volume , axis=0 , keepdims=True) * alpha25_vwap * (alpha25_high - alpha25_close))
    
    alpha = alpha25[0,:] * self.Universe_one.iloc[i ,:]
    
    return alpha


def modified_alpha26():
    # ################################################################################
    #Alpha26: (-1 * ts_max(correlation(ts_rank(volume , 5) , ts_rank(high , 5) , 5) , 3))
    # 含义: 买卖策略 买卖反向后
    # 买入 过去11天内 volume 和 high 相关性最低的股票，卖出相关性最高的股票 
    # 
    # 关键字：量价关系
    
    # 回测结果 持续下行 2017与市场走势较为一致
    
    ##cum_return_rate          0.823183
    ##final_return_rate         0.13254
    ##beta                     0.944506
    ##alpha                   0.0342594
    ##sharpe ratio             0.271224
    ##information ratio        0.111912
    ##turnover rate            0.070497
    ##max drawdown             0.566796
        ##drawdown start date  2017-12-25
        ##drawdown end date          None
    ##fitness                  0.371892
    # ################################################################################    
    d = 5 + (5 - 1) + (3 - 1)  # 共取了11天的数据
    alpha26_volume = self.Volume[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha26_adjhigh = self.Highprice[di-DELAY - d + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1:di-DELAY + 1 , :]
    rank_volume = np.zeros((7 , alpha26_volume.shape[1]))
    rank_high = np.zeros((7 , alpha26_volume.shape[1]))
    for ii in range(7):
        jj = ii + 5
        rank_volume[ii] = self._ts_rank(alpha26_volume[ii:jj , :])
        rank_high[ii] = self._ts_rank(alpha26_adjhigh[ii:jj , :])
    
    corr = np.zeros((3 , alpha26_volume.shape[1]))
    for ii in range(3):
        jj = ii + 5
        corr[ii] = self._correlation(rank_volume[ii:jj , :] , rank_high[ii:jj , :])
    alpha26 = -1 * np.nanmax(corr , axis=0)
    
    alpha26 = -1 * alpha26
    alpha = alpha26.reshape(alpha26_volume.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha 


def modified_alpha27():
    # ################################################################################
    # Alpha27 =((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    # 含义
    # 买卖策略
    # 当 0.5 < rank (排名靠后-数值较小：volume和vwap的相关系数) -- > 买入（原策略为卖出）
    # 当 0.5 < rank (排名靠后-数值较小：volume和vwap的相关系数) -- > 卖出（原策略为买入）
    # ----------------------------------
    # 买入 相关性较小的股票，卖出相关性较大的股票
    # 关键字：量价关系
    # 回测结果 买卖反向
    #
    # # ####
    # cum_return_rate           1.32174
    # final_return_rate        0.190721
    # beta                     0.910695
    # alpha                   0.0946882
    # sharpe ratio             0.440392
    # information ratio        0.324346
    # turnover rate                   0
    # max drawdown             0.523545
    #   drawdown start date  2017-12-05
    #   drawdown end date          None
    # fitness                       inf
    # ################################################################################        
    d = 7
    alpha27_volume = self.Volume[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha27_vwap = self.vwap[di-DELAY - d + 1:di-DELAY + 1 , :]
    corr = np.zeros((2 , alpha27_volume.shape[1]))  # 2行N列的矩阵，用于存放 Correlation结果
    
    for ii in range(2):
        jj = ii + 6
        corr[ii] = self._correlation(self._rank(alpha27_volume[ii:jj , :]) , self._rank(alpha27_vwap[ii:jj , :]))
    
    alpha27_cond = 0.5 < self._rank(np.nanmean(corr , axis=0 , keepdims=True))
    alpha27 = np.where(alpha27_cond , np.ones((1 , alpha27_volume.shape[1])) ,
                       (np.ones((1 , alpha27_volume.shape[1])) * -1))
    
    alpha27 = -1 * alpha27
    alpha = alpha27.reshape((alpha27_volume.shape[1] ,)) * self.Universe_one.iloc[i , :]
    
    return alpha


def modified_alpha28():
    # ################################################################################
    #Alpha28 = scale(((correlation(adv20 , low , 5) + ((high + low) / 2)) - close))
    # 含义
    # 相当于是1天内的reversal，
    # 买入 收盘价<均价 的 股票 如果过去一段时间 最低价和成交量的相关系数的正相关性很高，买入的比重更大
    # 卖出 收盘价>均价 的 股票 如果过去一段时间 最低价和成交量的相关系数的负相关性很高，买入的比重更大
    #
    # 逻辑 1天内的 reversal,同时根据量价关系调整weight 
      # 但是这里 相关性和买入卖出的逻辑 不太对 而且 1天内可能更多的是momentum
    
    # 关键字  Reversal； 量价走势
    # 回测结果 2015前温和上涨；2015暴涨暴跌；2016震荡；2017下行 
    #
    # cum_return_rate          0.988285
    # final_return_rate         0.15307
    # beta                     0.991947
    # alpha                   0.0516346
    # sharpe ratio             0.286426
    # information ratio        0.154502
    # turnover rate           0.0254813
    # max drawdown             0.616164
    #   drawdown start date  2017-12-05
    #   drawdown end date          None
    # fitness                  0.702016
    # ################################################################################
    
    d1 = 5
    d2 = 20 + (5 - 1)
    
    alpha28_high = self.Highprice[di-DELAY:di-DELAY + 1 , :]
    alpha28_low_range = self.Lowprice[di-DELAY - d1 + 1:di-DELAY + 1 , :]
    alpha28_close = self.Closeprice[di-DELAY:di-DELAY + 1 , :]
    alpha28_volume_range = (self.vwap[di-DELAY - d2 + 1:di-DELAY + 1 , :])
    
    alpha28_adv20 = np.zeros((d1 , alpha28_high.shape[1]))
    for ii in range(d1):
        alpha28_adv20[ii] = np.nanmean(alpha28_volume_range[ii:ii + 20 , :] , axis=0 , keepdims=True)
    
    alpha28_corr = self._correlation(alpha28_adv20 , alpha28_low_range)
    alpha28 = self._scale(alpha28_corr + alpha28_high + (alpha28_low_range[d1 - 1:d1 , :]) / 2 - alpha28_close)
    
    alpha = alpha28.reshape((alpha28_high.shape[1] ,)) * self.Universe_one.iloc[i , :]

    return alpha 




def modified_alpha29():
    # ################################################################################
    # alpha29 = (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    # # 逻辑:买入当天 close大幅上升 以及6天前return 处于历史高位 
    #  不太明白逻辑
    # 关键字：Momentum 
    
    # 回测结果 2015前温和上涨；2015暴涨暴跌；2016震荡；2017下行 
    ##cum_return_rate           1.08954
    ##final_return_rate        0.165001
    ##beta                     0.874385
    ##alpha                   0.0713831
    ##sharpe ratio             0.385689
    ##information ratio         0.24214
    ##turnover rate            0.127837
    ##max drawdown             0.548233
        ##drawdown start date  2017-12-25
        ##drawdown end date          None
    ##fitness                   0.43818
    # ################################################################################
    d1 = 5 
    d2 = 2 
    d3 = 2
    d4 = 2
    d5 = 5 

    d6 = 6 
    d7 = 5 

    alpha29_closeadj = self.Closeprice[di-DELAY-(d1+d2-1+d3-1+d4-1+d5-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1+d3-1+d4-1+d5-1)+1:di-DELAY+1,:]
    alpha29_return = np.diff(alpha29_closeadj[-(d6+d7-1+1):,:],axis = 0) / alpha29_closeadj[-(d6+d7):-1,:]

    # min(...)
    alpha29_min_delta = np.zeros((d2+d3-1+d4-1+d5-1, alpha29_closeadj.shape[1]))
    for ii in range(d2+d3-1+d4-1+d5-1):
        jj = ii + d1 
        alpha29_min_delta[ii] = self._delta(alpha29_closeadj[ii:jj]) # 是否减1没有影响

    alpha29_min_tsmin = np.zeros((d3+d4-1+d5-1, alpha29_closeadj.shape[1]))
    for ii in range(d3+d4-1+d5-1):
        jj = ii + d2 
        alpha29_min_tsmin[ii] = self._ts_min(self._rank(self._rank(-1*self._rank(alpha29_min_delta[ii:jj]))))

    alpha29_min_tsrank_sum = np.zeros((d4+d5-1, alpha29_closeadj.shape[1]))
    for ii in range(d4+d5-1):
        jj = ii + d3 
        alpha29_min_tsrank_sum[ii] = np.nanmean(alpha29_min_tsmin[ii:jj], axis=0, keepdims=True) * d3

    alpha29_min_product = np.zeros((d5, alpha29_closeadj.shape[1]))
    for ii in range(d5):
        jj = ii + d4 
        alpha29_min_product[ii] = np.nanprod(alpha29_min_tsrank_sum[ii:jj], axis=0, keepdims=True)

    alpha29_min = self._ts_min(alpha29_min_product)

    # ts_rank(...)
    alpha29_tsrank = self._ts_rank(alpha29_return[:-(d6-1),:])

    alpha29 = alpha29_min + alpha29_tsrank
    alpha =  alpha29[0,:] * self.Universe_one.iloc[i,:]

    return alpha 





def modified_alpha30():
    
    # ################################################################################
    # alpha30 = (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    # 含义：买入策略
    # 买入 收盘价格连续三天上涨 的股票 最近5天交易量越大的 权重越大 
    # 关键字：Momentum；
    # 逻辑：交易量辅助判断下的3天Momentum策略
    # 可能的问题： 这里通过交易量高辅助确认 momentum策略 --> 这里可能不对，可能需要用反过来 交易量较低 作为 momentum持续的一个逻辑
    #            或者在叠加一个量价相关性 
    
    # 回测结果 2015年以前温和上涨；2015暴涨暴跌；2016震荡；2017下行
    
    ##p_mean                    0.253728
    ##t_mean                     2.02573
    ##t_std                      2.67436
    ##t_sig_prt                 0.386326
    ##t_2                       -5.74259
    ##p_2                    1.04891e-08
    ##R_squared_adj_mean        0.026499
    ##R_squared_adj_std       -0.0152099
    ##IC_mean                  0.0446248
    ##IC_abs_mean               0.165569
    ##IC_std                    0.125647
    ##IC_positive_prt           0.423394
    ##IR                      -0.0918643
    # ################################################################################
    
    d1 = 4 
    d2 = 5 
    d3 = 20 

    alpha30_closeadj = self.Closeprice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
    alpha30_volume = self.Volume[di-DELAY-(d2+d3)+1:di-DELAY+1,:]

    alpha30_rank = self._rank(np.sign(alpha30_closeadj[d1-1:d1,:] - alpha30_closeadj[d1-2:d1-1,:]) + np.sign(alpha30_closeadj[d1-2:d1-1,:] - alpha30_closeadj[d1-3:d1-2,:]) + np.sign(alpha30_closeadj[d1-3:d1-2,:] - alpha30_closeadj[d1-4:d1-3,:]))

    alpha30_vol_ratio = np.nanmean(alpha30_volume[-d2:,:], axis=0, keepdims=True) * 5 / (np.nanmean(alpha30_volume[:-d2,:]) * 20)

    alpha30 = (1- alpha30_rank) * alpha30_vol_ratio
    alpha =  alpha30[0,:] * self.Universe_one.iloc[i,:]

    return alpha 













def modified_alpha31():
    
    # ################################################################################
    # alpha31 = ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    # 含义 买卖策略 
    # 买入：量价正相关 close相对于10天前大幅上升（且在过去10天内处于高位）+ close相对于3天前大幅上升；-- 权重越大 
    # 卖出：量价负相关 close相对于0天前大幅上升（且在过去10天内处于高位）+ close相对于3天前大幅上升；-- 权重越大
    
    # 关键字：量价关系  Momentum/reversal 
    # 长期和短期 close都处于较高的水平 --> 保持momentum 
    # 逻辑: 量价正相关 更容易保持 在价格高位上保持Momentum；量价负相关，更容易在价格高位上出现Reversal.
    # 
    # 回测结果 2015年以前温和上涨；2015暴涨暴跌；2016震荡；2017下行
    ##cum_return_rate           1.04219
    ##final_return_rate         0.15948
    ##beta                     0.948135
    ##alpha                   0.0609578
    ##sharpe ratio             0.336047
    ##information ratio        0.200504
    ##turnover rate            0.188166
    ##max drawdown             0.552684
        ##drawdown start date  2017-12-05
        ##drawdown end date          None
    ##fitness                  0.309373
    # ################################################################################
    
    # 三层 rank 函数其实只有1层的效果
    d1 = 10 
    d2 = 10 
    d0 = 20 
    d3 = 12
    alpha31_closeadj = self.Closeprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]
    alpha31_lowadj = self.Lowprice[di-DELAY-(d3)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3)+1:di-DELAY+1,:]
    alpha31_volume = self.Volume[di-DELAY-(d0+d3-1)+1:di-DELAY+1]

    alpha31_rank1_delta = np.diff(alpha31_closeadj, n=10, axis=0)
    alpha31_rank1 = self._rank(self._decay_linear(-1 * self._rank(self._rank(alpha31_rank1_delta))))

    alpha31_rank2 = self._rank((-1* np.diff(alpha31_closeadj, n=3, axis=0)[-1,:]).reshape(1,alpha31_closeadj.shape[1]))

    alpha31_sign_adv20 = np.zeros((d3, alpha31_closeadj.shape[1]))
    for ii in range(d3):
        jj = ii + d0 
        alpha31_sign_adv20[ii] = np.nanmean(alpha31_volume[ii:jj], axis=0, keepdims=True)

    alpha31_sign = np.sign(self._scale(self._correlation(alpha31_sign_adv20, alpha31_lowadj)))

    alpha31 = alpha31_rank1 + alpha31_rank2 + alpha31_sign 

    alpha = alpha31[0,:] * self.Universe_one.iloc[i,:]

    return alpha 





def modified_alpha32():
    # ################################################################################
    #alpha32 = (scale(((sum(close , 7) / 7) - close)) + (20 * scale(correlation(vwap , delay(close , 5) , 230))))
    # 成分1：过去7日收盘均价-今日收盘价； 成分2; 均价和一周前的收盘价的相关性
    # 
    # momentum策略 （相比过去7天的均值）-- 牛市挣钱 震荡市不行
    # 买入：昨日收盘 > 过去7天的收盘均价 且 (权重较大) 过去1年内滞后5天的收盘价格和日均价格正相关性较高的股票
    # 卖出：昨日收盘 < 过去7天的收盘均价 且 过去1年内滞后5天的收盘价格收盘价格和日均价格负相关性较高的股票
    
    # 逻辑：通过价格和滞后5天价格的相关性 来确认 Momentum/Reversal的存在 
    #       * 相关性越正下 如果 收盘价 < 过去7天均值则买入， 相反则卖出 --> Reversal; 如果相关性越负 收盘价 > 过去7天均值 （买入）--> Momentum 
    # 量价相关性 
    # 关键词：Momentum/Reversal；量价相关性 
    # 
    # 测试一下 如果只用第二部分的结果如何？
    
    # 回测结果 主要是2017年以前增长较大，2017年下行 -- 反向？
    
    # cum_return_rate            1.93868
    # final_return_rate         0.250312
    # beta                      0.892384
    # alpha                     0.155497
    # sharpe ratio              0.622742
    # information ratio         0.553522
    # turnover rate          0.000305819
    # max drawdown              0.499581
    #   drawdown start date   2015-09-15
    #   drawdown end date           None
    # fitness                    17.8163
    
    # ################################################################################    
    d1 = 7
    d2 = 235
    d3 = 5
    alpha32_close = self.Closeprice[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    alpha32_vwap = (self.vwap[di-DELAY - d2 + 1:di-DELAY + 1 , :])
    
    alpha32_1 = self._scale(np.nanmean(alpha32_close[-d1: , :] , axis=0 , keepdims=True) - alpha32_close[d2 - 1:d2 , :])
    alpha32_2 = 20 + self._scale(self._correlation(alpha32_vwap[d3: , :] , alpha32_close[:-d3 , :]))
    alpha32 = alpha32_1 + alpha32_2
    
    alpha = alpha32.reshape(alpha32_close.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha 





def modified_alpha33():
    
    # ################################################################################
    # alpha33 = rank((-1 * ((1 - (open / close))^1)))
    #
    # 含义 单日的Momentum策略
    # rank 为降序排列
    # 买入策略： 重仓买入 (close-open)/close 较大（正）的 股票
    # 关键词 Momentum 
    # 回测结果 2015年以前温和上涨；2015暴涨暴跌；2016震荡；2017下行
    
    # cum_return_rate           0.30403
    # final_return_rate       0.0565543
    # beta                     0.893956
    # alpha                   -0.038365
    # sharpe ratio            0.0603099
    # information ratio        -0.16604
    # turnover rate            0.264297
    # max drawdown              0.60964
    #   drawdown start date  2017-12-25
    #   drawdown end date          None
    # fitness                 0.0278982
    # ################################################################################
    alpha33_open = self.Openprice[di-DELAY:di-DELAY + 1]
    alpha33_close = self.Closeprice[di-DELAY:di-DELAY + 1]
    
    alpha33 = self._rank(-1 * (1 - alpha33_open / alpha33_close))
    
    alpha = alpha33.reshape(alpha33_open.shape[1] , ) * self.Universe_one.iloc[i , :]
    
    return alpha 







def modified_alpha34():
    
    # ################################################################################
    # alpha34 = rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    # 含义： 买入策略 买入 (1) 2天return波动/5天return波动较低 + (2) close下跌 
    # 逻辑：低波动率下的 reversal 
    # 关键词：Reversal; 收益率波动 
    # 回测结果 2016运行平稳，2017下行
    
    # cum_return_rate           1.79155
    # final_return_rate        0.237074
    # beta                     0.845889
    # alpha                    0.145351
    # sharpe ratio              0.62648
    # information ratio        0.542774
    # turnover rate              0.2036
    # max drawdown              0.47655
    #   drawdown start date  2017-12-25
    #   drawdown end date          None
    # fitness                  0.676021
    # ################################################################################
    
    d2 = 5 + 1  # 因为有return的数据所以多取一天
    
    alpha34_closeadj = self.Closeprice[di-DELAY - d2 + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    alpha34_return = np.diff(alpha34_closeadj , axis=0) / alpha34_closeadj[:-1]
    
    # 这里可能会存在问题，如果两天中有一天不存在，那么nanstd(return , 2) 返回的将会是0 所以改成 np.std
    alpha34_1 = 1 - self._rank(np.std(alpha34_return[-2: , :] , axis=0 , keepdims=True) / self._rank(np.nanstd(alpha34_return , axis=0 , keepdims=True)))
    alpha34_2 = 1 - self._rank(alpha34_closeadj[d2 - 1:d2 , :] - alpha34_closeadj[d2 - 2:d2 - 1 , :])
    alpha34 = self._rank(alpha34_1 + alpha34_2)
    
    alpha = alpha34.reshape(alpha34_closeadj.shape[1] , ) * self.Universe_one.iloc[i , :]
    return alpha 





def modified_alpha35():

    
    # ################################################################################
    # alpha35 = ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
    # 含义：交易量处于低位+价格处于高位+return处于高位 --> 大量买  Momentum策略 16/32天周期
    # 逻辑：高return+低交易量 确认Momentum存在  
    #    * 问题： 2016年以后 16/32天内可能Reversal更好 
    #    *       对于 close+high-low的含义不太明确 --> 
    # 关键词：Momentum+量、收益率辅助确认
    # 回测结果 2015年以前温和上涨；2015暴涨暴跌；2016震荡；2017下行；另有大量超配 > 5%
    
    # cum_return_rate          0.128353
    # final_return_rate       0.0253414
    # beta                     0.872364
    # alpha                   -0.068142
    # sharpe ratio            -0.028662
    # information ratio       -0.270727
    # turnover rate            0.327671
    # max drawdown             0.623411
    #   drawdown start date  2017-12-25
    #   drawdown end date          None
    # fitness               -0.00797081
    # ################################################################################
    
    d1 = 32
    d2 = d1 + 1  # return 需要多取一天的数值
    d3 = 16
    
    alpha35_volume = self.Volume[di-DELAY - d1 + 1: di-DELAY + 1 , :]
    alpha35_closeadj = self.Closeprice[di-DELAY - d2 + 1: di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d2 + 1: di-DELAY + 1 , :]
    alpha35_highadj = self.Highprice[di-DELAY - d3 + 1: di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d3 + 1: di-DELAY + 1 , :]
    alpha35_lowadj = self.Lowprice[di-DELAY - d3 + 1: di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d3 + 1: di-DELAY + 1 , :]
    
    alpha35_return = np.diff(alpha35_closeadj , axis=0) / alpha35_closeadj[:-1]
    
    # return 32x N; close 33xN; high/low: 16xN; volume 32xN
    volume_rank = self._ts_rank(alpha35_volume)
    price_rank = 1 - self._ts_rank(alpha35_closeadj[-16: , :] + alpha35_highadj - alpha35_lowadj)
    return_rank = 1 - self._ts_rank(alpha35_return)
    
    alpha35 = volume_rank * price_rank * return_rank
    alpha = alpha35.reshape(alpha35_closeadj.shape[1] , ) * self.Universe_one.iloc[i , :]
    
    return alpha 



def modified_alpha36_totest():
    
    # ################################################################################
    # alpha36 = (
    #             (
    #               (
    #                  ((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) 
    #                 + (0.7 * rank((open - close)))) 
    #                 +(0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))
    #                ) 
    #               + rank(abs(correlation(vwap, adv20, 6)))
    #             )
    #             +(0.6 * rank((((sum(close, 200) / 200) - open) * (close - open))))
    #           )
    
    # 含义：买入策略
    #   当日涨幅和前一天交易量的相关性较低 
    #   当日下跌较多
    #   1周前的return再过去1周内处于历史低位
    #   价和平均成交量的相关系数绝对值较小 
    #   短期和长期价格走势出现背离 当日收盘开盘价和当日200天均价 分别在收盘价差两侧 （如果当天收盘涨，相对于基准期就是跌；如果当天跌，相对于长期就是涨）
    # 逻辑：Momentum/Reversal 
    #   问题：可否再次基础上做一些简化，比如单独看第一个和最后一个说不定也是比较好的反转信号
    #   
    # 回测结果  2015年以前温和上涨；2015暴涨暴跌；2016震荡；2017下行
    ##cum_return_rate          0.777832
    ##final_return_rate        0.126644
    ##beta                      0.95951
    ##alpha                    0.027365
    ##sharpe ratio             0.235454
    ##information ratio       0.0801924
    ##turnover rate           0.0562889
    ##max drawdown             0.577964
        ##drawdown start date  2017-12-05
        ##drawdown end date          None
    ##fitness                  0.353172
    # ################################################################################
    
    
    d1 = 200 
    d2 = 15
    d0 = 20 
    d3 = 6 

    alpha36_close = self.Closeprice[di-DELAY:di-DELAY+1,:] 
    alpha36_open = self.Openprice[di-DELAY:di-DELAY+1,:] 
    alpha36_closeadj = self.Closeprice[di-DELAY-d2+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d2+1:di-DELAY+1,:]
    alpha36_openadj = self.Openprice[di-DELAY-d2+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d2+1:di-DELAY+1,:]
    alpha36_vwapadj = self.vwap[di-DELAY-(d3)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3)+1:di-DELAY+1,:]


    alpha36_volume = self.Volume[di-DELAY-d1+1:di-DELAY+1,:]

    alpha36_rank1 = self._rank(self._correlation((alpha36_closeadj-alpha36_openadj), alpha36_volume[-16:-1,:]))

    alpha36_rank2 = self._rank(alpha36_open - alpha36_close)


    alpha36_return = np.diff(alpha36_closeadj, axis=0) / alpha36_closeadj[:-1]
    alpha36_rank3 = self._rank(self._ts_rank(alpha36_return[-11:-6]))

    alpha36_rank4_adv20 = np.zeros((d3, alpha36_close.shape[1]))
    for ii in range(d3):
        jj = ii + d0 
        alpha36_rank4_adv20[ii] = np.nanmean((alpha36_volume[-(d0+d3-1):,:])[ii:jj], axis=0, keepdims=True)

    alpha36_rank4 = self._rank(np.abs(self._correlation(alpha36_vwapadj, alpha36_rank4_adv20)))


    alpha36_rank5 = self._rank((np.nanmean(alpha36_closeadj, axis=0, keepdims=True) - alpha36_open) *(alpha36_close-alpha36_open) )

    alpha36 = (2.21 * alpha36_rank1 + 0.7 * alpha36_rank2 + 0.73 * alpha36_rank3 + 0.6 * alpha36_rank4 +0.6 * alpha36_rank5 )

    alpha = alpha36[0,:] * self.Universe_one.iloc[i,:]

    return alpha 








def modified_alpha37():
    
    # ################################################################################
    # alpha37 =  (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    # 含义：买入策略， 买入 前一日涨幅（close-open) 和当日收盘价的相关性越正+ 当日 涨幅较大的股票 
    # 逻辑：1天的 Momentum策略，前日涨幅和今日收盘价的相关性 辅助确认 Momentum的存在 
    #  
    # 关键词：Momentum  价格走势辅助判断
    # 回测结果  2015年以前温和上涨；2015暴涨暴跌；2016震荡；2017下行
    
    # cum_return_rate            1.2383
    # final_return_rate        0.181723
    # beta                     0.896703
    # alpha                   0.0866214
    # sharpe ratio             0.419505
    # information ratio        0.293522
    # turnover rate            0.127007
    # max drawdown             0.523459
    #   drawdown start date  2015-09-01
    #   drawdown end date          None
    # fitness                  0.501797
    # ################################################################################
    
    d1 = 200
    d2 = d1 + 1
    
    alpha37_openadj = self.Openprice[di-DELAY - d2 + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    alpha37_closeadj = self.Closeprice[di-DELAY - d2 + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    alpha37_open = self.Openprice[di-DELAY:di-DELAY + 1]
    alpha37_close = self.Closeprice[di-DELAY:di-DELAY + 1]
    
    # opanadj/closeadj: 201xN; open/close: 1xN
    rank_corr = self._rank(self._correlation((alpha37_openadj - alpha37_closeadj)[:-1] , alpha37_closeadj[1: , :]))
    rank_prcdiff = self._rank(alpha37_open - alpha37_close)
    alpha37 = rank_corr + rank_prcdiff
    
    alpha = alpha37.reshape(alpha37_close.shape[1] , ) * self.Universe_one.iloc[i , :]
    return alpha     





def modified_alpha38():
    
    # ################################################################################
    # alpha38 = ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    
    # 买卖反向后 
    # 含义
    #
    
    # 回测结果 （2017年之后开始下跌) -- 反向
    # cum_return_rate           2.35146
    # final_return_rate        0.284837
    # beta                     0.987556
    # alpha                    0.183693
    # sharpe ratio             0.649505
    # information ratio        0.613845
    # turnover rate            0.356354
    # max drawdown             0.545028
    #   drawdown start date  2015-09-15
    #   drawdown end date          None
    # fitness                  0.580684
    # ################################################################################    
    d = 10
    alpha38_closeadj = self.Openprice[di-DELAY - d + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha38_close = self.Closeprice[di-DELAY:di-DELAY + 1 , :]
    alpha38_open = self.Openprice[di-DELAY:di-DELAY + 1 , :]
    rank1 = -1 * self._rank(self._ts_rank(alpha38_closeadj))
    rank2 = self._rank(alpha38_close / alpha38_open)
    
    alpha38 = rank1 * rank2
    # 测试买卖反向
    alpha38 = -1 * alpha38
    alpha = alpha38.reshape(alpha38_close.shape[1] , ) * self.Universe_one.iloc[i , :]
    return alpha


# ################################################################################
# alpha39 = ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))

# 回测结果 主要上涨在2015年，2017年呈下降趋势
#
# cum_return_rate           1.81319
# final_return_rate        0.239055
# beta                      1.02349
# alpha                    0.135522
# sharpe ratio             0.525915
# information ratio        0.465118
# turnover rate           0.0944283
# max drawdown             0.547346
#   drawdown start date  2015-09-01
#   drawdown end date          None
# fitness                  0.836783

# ################################################################################
def modified_alpha39():
    d1 = 20
    d2 = 9
    d3 = 250 + 1  # 由于要计算return ， closeadj 取251天
    alpha39_closeadj = self.Closeprice[di-DELAY - d3 + 1:di-DELAY + 1] * self.adjfactor[di-DELAY - d3 + 1:di-DELAY + 1]
    alpha39_volume = self.Volume[di-DELAY - d2 + 1:di-DELAY + 1 , :]
    
    alpha39_adv20 = np.zeros((d2 , alpha39_volume.shape[1]))  # 用于存放adv20
    for ii in range(d2):
        jj = ii + d1
        alpha39_adv20[ii] = np.nanmean(alpha39_volume[ii:jj , :])
    
    alpha39_return = np.diff(alpha39_closeadj , axis=0) / alpha39_closeadj[:-1]
    rank1 = -1 * self._rank(self._delta(alpha39_closeadj[-8: , :]))
    rank2 = 1 - self._rank(self._decay_linear(alpha39_volume / alpha39_adv20))
    rank3 = 1 + self._rank(self._ts_sum(alpha39_return))
    
    alpha39 = rank1 * rank2 * rank3
    # 测试买卖反向
    alpha39 = -1 * alpha39
    alpha = alpha39.reshape(alpha39_volume.shape[1] , ) * user.Universe_one.iloc[i , :]

    return alpha 
# ################################################################################
# alpha40 = ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))

# 买卖反向
# 回测结果 2015年未能上涨 近两年走势平稳（但增长不大）
# cum_return_rate        -0.0806396
# final_return_rate      -0.0172729
# beta                    -0.259922
# alpha                  -0.0354592
# sharpe ratio            -0.261429
# information ratio       -0.327888
# turnover rate            0.142003
# max drawdown             0.457191
#   drawdown start date  2016-01-13
#   drawdown end date          None
# fitness                -0.0911778
# ################################################################################
def modified_alpha40():
    d = 10
    
    alpha40_highadj = self.Highprice[di-DELAY - d + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha40_volume = self.Volume[di-DELAY - d + 1:di-DELAY + 1 , :]
    
    alpha40_rank = -1 * self._rank(np.nanstd(alpha40_highadj , axis=0 , keepdims=True))
    alpha40_corr = self._correlation(alpha40_highadj , alpha40_volume)
    
    alpha40 = alpha40_rank + alpha40_corr
    
    # 买卖反向
    alpha40 = alpha40 * -1
    alpha = alpha40.reshape(alpha40_volume.shape[1] , ) * self.Universe_one.iloc[i , :]
    
    return alpha 

# ################################################################################
# alpha41 = (((high * low)^0.5) - vwap)

# 回测结果 2015年大涨 到5倍左右 2017年下降
#
# cum_return_rate           1.28552
# final_return_rate        0.186847
# beta                      1.04221
# alpha                   0.0820694
# sharpe ratio             0.351571
# information ratio        0.244861
# turnover rate            0.170774
# max drawdown             0.615082
#   drawdown start date  2017-12-25
#   drawdown end date          None
# fitness                  0.367744
# ################################################################################

def modified_alpha41():
        
    alpha41_high = self.Highprice[di-DELAY:di-DELAY + 1 , :]
    alpha41_vwap = self.vwap[di-DELAY:di-DELAY + 1 , :]
    
    alpha41 = np.sqrt(alpha41_high * alpha41_vwap) - alpha41_vwap
    alpha = alpha41.reshape(alpha41_high.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha 

# ################################################################################
# alpha42 = (rank((vwap - close)) / rank((vwap + close)))

# 买卖反向
# 回测结果 2015年大涨阶段下跌，今年表现与较好 但在很多个股上超配较为严重
# cum_return_rate          0.334799
# final_return_rate       0.0616731
# beta                    -0.695441
# alpha                   0.0724488
# sharpe ratio            0.0669157
# information ratio      -0.0739048
# turnover rate            0.425979
# max drawdown             0.722989
#   drawdown start date  2015-06-03
#   drawdown end date    2017-01-16
# fitness                 0.0254614
# ################################################################################
def modified_alpha42():
    
    alpha42_vwap = self.vwap[di-DELAY:di-DELAY + 1 , :]
    alpha42_close = self.Closeprice[di-DELAY:di-DELAY + 1 , :]
    
    alpha42 = self._rank(alpha42_vwap - alpha42_close) / self._rank(alpha42_vwap + alpha42_close)
    
    alpha = alpha42.reshape(alpha42_close.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha 

# ################################################################################
# alpha43 =  (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))

# 回测结果 主要涨幅在2015 2016， 2017年下行

# cum_return_rate           1.11323
# final_return_rate        0.167726
# beta                     0.896919
# alpha                   0.0726096
# sharpe ratio             0.377292
# information ratio        0.239892
# turnover rate            0.196357
# max drawdown             0.473255
#   drawdown start date  2017-12-05
#   drawdown end date          None
# fitness                  0.348702
# ################################################################################
def modified_alpha43():
    
    d1 = 20
    d2 = 20
    d3 = 7
    d4 = 8
    
    alpha43_volume = self.Volume[di-DELAY - d1 - (d2 - 1) + 1:di-DELAY + 1 , :]
    alpha43_closeadj = self.Closeprice[di-DELAY-(d3+d4-1) + 1:di-DELAY+1, :] * self.adjfactor[di-DELAY-(d3+d4-1) + 1:di-DELAY+1, :]
    alpha43_adv20 = np.zeros((d1 , alpha43_volume.shape[1]))
    for ii in range(d1):
        jj = ii + d2
        alpha43_adv20[ii] = np.nanmean(alpha43_volume[ii:jj , :] , axis=0 , keepdims=True)
    
    alpha43_lag_close = np.zeros((d4 , alpha43_volume.shape[1]))
    for ii in range(d4):
        jj = ii + d3
        alpha43_lag_close[ii] = self._delta(alpha43_closeadj[ii:jj , :])
    
    alpha43 = self._ts_rank(alpha43_volume[-20: , :] / alpha43_adv20) * self._ts_rank(-1 * alpha43_lag_close)
    
    alpha = alpha43.reshape(alpha43_volume.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha 
# ################################################################################
# alpha44 = (-1 * correlation(high, rank(volume), 5))

# 回测结果 近一年下行，2015和2016表现还可以
# cum_return_rate          0.398907
# final_return_rate       0.0720445
# beta                     0.657724
# alpha                 -0.00716534
# sharpe ratio             0.141227
# information ratio       -0.136422
# turnover rate            0.187341
# max drawdown             0.512364
#   drawdown start date  2017-12-25
#   drawdown end date          None
# fitness                 0.0875794
# ################################################################################
def modified_alpha44():
    d = 5
    alpha44_volume = self.Volume[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha44_highadj = self.Highprice[di-DELAY - d + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1:di-DELAY + 1 , :]

    alpha44 = -1 * self._correlation(alpha44_highadj , self._rank(alpha44_volume))

    alpha = alpha44.reshape(alpha44_volume.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha 




# ################################################################################
# alpha45 = (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))

# 回测结果  牛熊市反向
# cum_return_rate         -0.105025
# final_return_rate      -0.0227326
# beta                    -0.259094
# alpha                   -0.040974
# sharpe ratio            -0.311822
# information ratio       -0.350903
# turnover rate            0.430141
# max drawdown             0.428482
#   drawdown start date  2015-08-17
#   drawdown end date          None
# fitness                -0.0716845
# ################################################################################

def modified_alpha45():
    d1 = 5
    d2 = 20
    d3 = 2

    alpha45_closeadj = self.Closeprice[di-DELAY - d2 - (d1 - 1) + 1: di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d2 - (d1 - 1) + 1: di-DELAY + 1 ,:]
    alpha45_volume = self.Volume[di-DELAY - d2 - (d1 - 1) + 1: di-DELAY + 1 , :]

    alpha45_1 = -1 * self._rank(np.nanmean(alpha45_closeadj[:-4 , :] , axis=0 , keepdims=True))
    alpha45_2 = self._correlation(alpha45_closeadj[-2: , :] , alpha45_volume[-2: , :])

    alpha45_sum_close5 = np.zeros((2 , alpha45_closeadj.shape[1]))
    alpha45_sum_close20 = np.zeros((2 , alpha45_closeadj.shape[1]))
    alpha45_sum_close5[0] = np.nansum(alpha45_closeadj[-6:-1 , :] , axis=0 , keepdims=True)
    alpha45_sum_close5[1] = np.nansum(alpha45_closeadj[-5: , :] , axis=0 , keepdims=True)
    alpha45_sum_close20[0] = np.nansum(alpha45_closeadj[-21:-1 , :] , axis=0 , keepdims=True)
    alpha45_sum_close20[1] = np.nansum(alpha45_closeadj[-20: , :] , axis=0 , keepdims=True)
    alpha45_3 = self._rank(self._correlation(alpha45_sum_close5 , alpha45_sum_close20))

    alpha45 = alpha45_1 * alpha45_2 * alpha45_3

    alpha = alpha45[0,:] * self.Universe_one.iloc[i , :]


    return alpha 



def modified_alpha46():
    
    
    # ################################################################################
    # alpha46 = ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
    # alpha46 = (cond1 ? (-1 * 1) : (cond2 ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
    
    # 回测结果 2015年以前温和上涨；2015暴涨暴跌；2016震荡；2017下行
    ##cum_return_rate          0.564547
    ##final_return_rate       0.0971967
    ##beta                     0.209249
    ##alpha                   0.0478105
    ##sharpe ratio             0.235568
    ##information ratio      -0.0148251
    ##turnover rate            0.228083
    ##max drawdown             0.402252
        ##drawdown start date  2015-07-08
        ##drawdown end date          None
    ##fitness                  0.153779
    # ################################################################################
        
    
    d1 = 20 + 1
    alpha46_closeadj = self.Closeprice[di-DELAY-(d1)+1:di-DELAY+1,:]

    alpha46_cond1 = ((alpha46_closeadj[0,:] - alpha46_closeadj[10,:])/10 - (alpha46_closeadj[10,:] - alpha46_closeadj[-1,:])/10 ) > 0.25
    alpha46_cond2 = ((alpha46_closeadj[0,:] - alpha46_closeadj[10,:])/10 - (alpha46_closeadj[10,:] - alpha46_closeadj[-1,:])/10 ) < 0


    alpha46_value1 = -1 * np.ones((alpha46_closeadj.shape[1]))
    alpha46_value2 = np.ones((alpha46_closeadj.shape[1]))
    alpha46_value3 = -1 * (alpha46_closeadj[-1] - alpha46_closeadj[-2])

    alpha46 = np.where(alpha46_cond1, alpha46_value1, (np.where(alpha46_cond2, alpha46_value2, alpha46_value3))).reshape((1,alpha46_closeadj.shape[1]))
    alpha = alpha46[0,:] * self.Universe_one.iloc[i , :]

    return alpha





def modified_alpha47():
    
    
    # ################################################################################
    # alpha47 = ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
    
    # 回测结果 持续下行 考虑交换逻辑
    ##cum_return_rate          -0.85678
    ##final_return_rate       -0.331513
    ##beta                    -0.996522
    ##alpha                   -0.300715
    ##sharpe ratio            -0.948757
    ##information ratio        -0.75475
    ##turnover rate            0.115499
    ##max drawdown             0.901255
      ##drawdown start date  2016-11-22
      ##drawdown end date          None
    ##fitness                  -1.60737
    # ################################################################################
    
    d0 = 20 
    d1 = 5 

    alpha47_highadj = self.Highprice[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:]
    alpha47_vwapadj = self.vwap[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:]
    alpha47_high = self.Highprice[di-DELAY:di-DELAY+1,:]
    alpha47_close = self.Closeprice[di-DELAY:di-DELAY+1,:]
    alpha47_volume = self.Volume[di-DELAY-d0+1:di-DELAY+1,:]

    alpha47_adv20 = np.nanmean(alpha47_volume, axis=0, keepdims=True)

    alpha47_rank1 = self._rank((1/alpha47_close*alpha47_volume[d0-1:d0,:])/alpha47_adv20) * alpha47_high * self._rank(alpha47_high) / np.nanmean(alpha47_highadj)
    alpha47_rank2 = self._rank(alpha47_vwapadj[d1-1:d1,:] - alpha47_vwapadj[0:1,:])

    alpha47 = alpha47_rank1 - alpha47_rank2

    alpha = alpha47[0,:] * self.Universe_one.iloc[i,:]
    return alpha 








def modified_alpha48_totest():
    
    # ################################################################################
    # alpha48 =  (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    # alpha48 =  ( indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) 
    #              / 
    #              sum(((delta(close, 1) / delay(close, 1))^2), 250)
    #            )
    
    # 回测结果            
    # valify 验证通过 by WJY
    # ################################################################################

    d1 = 252 

    alpha48_closeadj = user.Closeprice[di-DELAY-(d1)+1:di-DELAY+1,:]
    alpha48_return = np.diff(alpha48_closeadj, axis=0) / alpha48_closeadj[:-1]
    Industry = user.Industry[di-DELAY:di-DELAY+1,:]


    alpha48_corr = user._correlation(np.diff(alpha48_closeadj, axis=0)[1:,:], np.diff(alpha48_closeadj, axis=0)[:-1,:])

    alpha48_Ind_corr = user._industry_neutral( alpha48_corr * (np.diff(alpha48_closeadj[-2:,:], axis=0)/alpha48_closeadj[251:252,:]), Industry )

    # alpha48_sum 本质上是 return 
    alpha48_sum = ((np.diff(alpha48_closeadj, axis=0) / alpha48_closeadj[:-1] )**2)

    alpha48 = alpha48_Ind_corr / alpha48_sum


    alpha = alpha48[0,:] * user.Universe_one.iloc[i,:]

    return alpha 








def modified_alpha49_totest():
    # ################################################################################
    # alpha49(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1)))) 
    
    # 回测结果
    ##cum_return_rate           0.81107
    ##final_return_rate        0.130977
    ##beta                     0.367373
    ##alpha                   0.0710753
    ##sharpe ratio             0.321981
    ##information ratio       0.0899415
    ##turnover rate            0.409065
    ##max drawdown             0.391812
        ##drawdown start date  2015-07-08
        ##drawdown end date    2016-07-19
    ##fitness                  0.182193
    # ################################################################################

    d1 = 20 + 1
    alpha49_closeadj = self.Closeprice[di-DELAY-(d1)+1:di-DELAY+1,:]

    alpha49_cond = ((alpha49_closeadj[0,:] - alpha49_closeadj[10,:])/10 - (alpha49_closeadj[10,:] - alpha49_closeadj[-1,:])/10 ) < -0.1

    alpha49 = np.where(alpha49_cond, np.ones((alpha49_closeadj.shape[1])), -1 * (alpha49_closeadj[-1] - alpha49_closeadj[-2]))

    alpha = alpha49[0,:] * self.Universe_one.iloc[i , :]
    return alpha 





# ################################################################################
# alpha50 =  (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))

# 回测结果
# 买卖反向
##cum_return_rate           1.35793
##final_return_rate        0.194543
##beta                      1.00907
##alpha                   0.0919692
##sharpe ratio             0.405236
##information ratio        0.303817
##turnover rate           0.0351475
##max drawdown             0.564598
  ##drawdown start date  2015-09-15
  ##drawdown end date          None
##fitness                  0.953387
# ################################################################################


# alpha50 =  (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))

def modified_alpha50():
    d = 9  # 共计取5+（5-1） = 9天的数据

    alpha50_volume = self.Volume[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha50_vwap = self.vwap[di-DELAY - d + 1:di-DELAY + 1 , :]

    alpha50_corr = np.zeros((5 , alpha50_volume.shape[1]))

    for ii in range(5):
        jj = ii + 5
        alpha50_corr[ii] = self._correlation(self._rank(alpha50_volume[ii:jj]) , self._rank(alpha50_vwap[ii:jj]))

    alpha50 = -1 * self._ts_max(self._rank(alpha50_corr))

    # 买卖反向
    alpha50 = alpha50 * -1
    alpha = alpha50.reshape(alpha50_volume.shape[1] , ) * self.Universe_one.iloc[i , :]
    return alpha 




def modified_alpha51():
    # ################################################################################
    # alpha51 = (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    
    # 回测结果 2015年以前温和上涨；2015暴涨暴跌；2016震荡；2017下行； 但最高涨幅不高
    ##cum_return_rate           0.403317
    ##final_return_rate         0.072744
    ##beta                      0.555279
    ##alpha                  0.000346835
    ##sharpe ratio              0.122667
    ##information ratio       -0.0998259
    ##turnover rate             0.291638
    ##max drawdown              0.462951
        ##drawdown start date   2017-12-25
        ##drawdown end date           None
    ##fitness                  0.0612639
    # ################################################################################    
    d1 = 20+1
    d2 = 10 

    alpha51_closeadj = self.Closeprice[di-DELAY-(d1)+1:di-DELAY+1, :]

    alpha51_cond = (alpha51_closeadj[0,:] + alpha51_closeadj[-1] - alpha51_closeadj[d2,:] * 2) < -0.05

    alpha51 = np.where(alpha51_cond, np.ones((alpha51_closeadj.shape[1])),alpha51_closeadj[-1,:] - alpha51_closeadj[-2,:] )

    # alpha51 已经是一个 array格式 的数据
    alpha = alpha51 * self.Universe_one.iloc[i,:]
    return alpha 


def modified_alpha52():
    
    # ################################################################################
    # alpha52 = ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    
    # 回测结果 2015年以前与大盘一致，2015以后与大盘完全反向
    # Valify 测试通过 by xx
    # ################################################################################
    
    d1 = 5 
    d2 = 5+1 # delay 5天 需要取 6天的数值 

    d3 = 240+1 # 计算return，price需要多取1天的数值
    d4 = 20+1 # 计算return，price需要多取1天的数值


    alpha52_lowadj = self.Lowprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactoer[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]

    alpha52_closeadj = self.Closeprice[di-DELAY-(d3)+1:di-DELAY+1, :] * self.adjfactor[di-DELAY-(d3)+1:di-DELAY+1, :]
    alpha52_return = np.diff(alpha52_closeadj)/ alpha52_closeadj[:-1]

    alpha52_volume = self.Lowprice[di-DELAY-(d1)+1:di-DELAY+1,:]

    alpha52_tsmin = self._ts_min(alpha52_lowadj[-5:])

    alpha52_delay_tsmin = np.zeros((d2,alpha52_lowadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha52_delay_tsmin[ii] = self._ts_min(alpha52_lowadj[ii:jj])

    alpha52_delay = alpha52_delay_tsmin[0:1,:]

    # 本质上  (sum(returns, 240) - sum(returns, 20)) / 220 是 20-240 天内的平均收益率
    alpha52_rank = self._rank(np.nanmean(alpha52_return[:-20], axis=0, keepdims=True))

    alpha52_tsrank = self._tsrank(alpha52_volume)

    alpha52 = (-1 * alpha52_tsmin + alpha52_delay) * alpha52_rank * alpha52_tsrank

    alpha = alpha52[0,:] * self.Universe_one.iloc[i,:]
    return alpha 



def modified_alpha53():
    # ################################################################################
    # alpha53 = (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    
    # 回测结果
    
    #结果较为奇怪：除了2015大牛市行情中，其余时间似乎仓位=0？
    
    ## cum_return_rate         0.00254809
    ## final_return_rate      0.000527526
    ## beta                    -0.0730535
    ## alpha                   -0.0300856
    ## sharpe ratio             -0.288489
    ## information ratio         -0.35059
    ## turnover rate            0.0661632
    ## max drawdown              0.320406
    ##   drawdown start date   2015-12-29
    ##   drawdown end date           None
    ## fitness                 -0.0257598
    # ################################################################################    
    d = 9 + 1
    
    alpha53_closeadj = self.Closeprice[di-DELAY - d + 1: di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha53_lowadj = self.Lowprice[di-DELAY - d + 1: di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha53_highadj = self.Highprice[di-DELAY - d + 1: di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1: di-DELAY + 1 , :]
    
    alpha53 = -1 * self._delta((alpha53_highadj - alpha53_lowadj) / (alpha53_closeadj - alpha53_lowadj))
    
    alpha = alpha53.reshape(alpha53.shape[1] , ) * self.Universe_one.iloc[i , :]
    
    return alpha 







def modified_alpha54():
    # ################################################################################
    # alpha54 = ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    
    # 回测结果
    # 近两年下行
    #
    # cum_return_rate          0.039658
    # final_return_rate      0.00809238
    # beta                       1.0099
    # alpha                   -0.094537
    # sharpe ratio           -0.0687444
    # information ratio       -0.301008
    # turnover rate            0.249106
    # max drawdown             0.682598
    #   drawdown start date  2017-12-25
    #   drawdown end date          None
    # fitness                -0.0123903
    # ################################################################################
    
    
    alpha54_low = self.Lowprice[di-DELAY:di-DELAY + 1 , :]
    alpha54_close = self.Closeprice[di-DELAY:di-DELAY + 1 , :]
    alpha54_open = self.Openprice[di-DELAY:di-DELAY + 1 , :]
    alpha54_high = self.Highprice[di-DELAY:di-DELAY + 1 , :]
    
    alpha54 = -1 * ((alpha54_low - alpha54_close) * alpha54_open ** 0.5) / ((alpha54_low - alpha54_high) * alpha54_close ** 0.5)
    
    # 买卖反向
    alpha54 = -1 * alpha54
    alpha = alpha54.reshape(alpha54_low.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha 

def modified_alpha55():
    # ################################################################################
    # alpha55 = (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
    
    
    # 回测结果 反向后 稳定在增长；但增长率不高
    ##cum_return_rate          0.409271
    ##final_return_rate       0.0736857
    ##beta                     0.148663
    ##alpha                   0.0283284
    ##sharpe ratio             0.358056
    ##information ratio       -0.122009
    ##turnover rate            0.250494
    ##max drawdown             0.117146
      ##drawdown start date  2015-07-07
      ##drawdown end date    2016-02-19
    ##fitness                  0.194198
    # ################################################################################    
    d = 12 + 6 - 1

    alpha55_closeadj = self.Closeprice[di-DELAY - d + 1: di-DELAY + 1 , :] * self.adjfactor[ di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha55_lowadj = self.Lowprice[di-DELAY - d + 1: di-DELAY + 1 , :] * self.adjfactor[ di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha55_highadj = self.Highprice[di-DELAY - d + 1: di-DELAY + 1 , :] * self.adjfactor[ di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha55_volume = self.Volume[di-DELAY - d + 1: di-DELAY + 1 , :]

    low_tsmin = np.zeros((6 , alpha55_closeadj.shape[1]))
    high_tsmax = np.zeros((6 , alpha55_closeadj.shape[1]))

    for ii in range(6):
        jj = ii + 12
        low_tsmin[ii] = self._ts_min(alpha55_lowadj[ii:jj , :])
        high_tsmax[ii] = self._ts_max(alpha55_highadj[ii:jj , :])

    alpha55 = -1 * self._correlation(self._rank((alpha55_closeadj[-6: , :] - low_tsmin) / high_tsmax - low_tsmin) , self._rank(alpha55_volume[-6: , :]))
    # alpha55 = -1 * alpha55
    alpha = alpha55.reshape(alpha55_closeadj.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha 

def modified_alpha56():

    
    
    # ################################################################################
    # alpha56 = (0 - (1 * (rank((sum(returns , 10) / sum(sum(returns , 2) , 3))) * rank((returns * cap)))))
    
    # 回测结果
    # 牛市较好，近一年走低
    # cum_return_rate           1.85812
    # final_return_rate         0.24313
    # beta                      1.06052
    # alpha                    0.137134
    # sharpe ratio             0.508638
    # information ratio        0.448099
    # turnover rate            0.308951
    # max drawdown             0.602931
    #   drawdown start date  2015-09-15
    #   drawdown end date          None
    # fitness                  0.451215
    # ################################################################################

    DELAY = self.DELAY
    d = 11

    alpha56_closeadj = self.Closeprice[di-DELAY - d + 1: di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha56_return = np.diff(alpha56_closeadj , axis=0) / alpha56_closeadj[:-1 , :]
    alpha56_cap = self.Value_mv[di-DELAY:di-DELAY + 1 , :]

    alpha56_return = np.nan_to_num(alpha56_return)
    return_ratio = np.nansum(alpha56_return , axis=0 , keepdims=True) / (np.nansum(alpha56_return[9:11 , :] , axis=0 , keepdims=True) + np.nansum(alpha56_return[8:10 , :] , axis=0 , keepdims=True) +np.nansum(alpha56_return[7:9 , :] , axis=0 , keepdims=True))

    alpha56 = (0 - self._rank(return_ratio) * self._rank(alpha56_return[9:10 , :] * alpha56_cap))
    # 买卖反向
    alpha56 = -1 * alpha56
    alpha = alpha56.reshape(alpha56_closeadj.shape[1] , ) * user.Universe_one.iloc[i , :]

    return alpha 









def modified_alpha57():

    
    # ################################################################################
    # alpha57 =  (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    
    
    # 回测结果
    # 很多股票超配（>5%) -- 数据较为异常，检查是否应用了未来函数
    
    # cum_return_rate           13.2057
    # final_return_rate        0.733136
    # beta                   -0.0621727
    # alpha                    0.701799
    # sharpe ratio              2.07636
    # information ratio         1.48503
    # turnover rate            0.608275
    # max drawdown             0.270862
    #   drawdown start date  2015-09-15
    #   drawdown end date    2015-12-31
    # fitness                   2.27953
    # ################################################################################

    DELAY = self.DELAY
    d = 31
    alpha57_closeadj = self.Closeprice[di-DELAY - d + 1: di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1: di-DELAY + 1 , :]
    alpha57_close = self.Closeprice[di-DELAY: di-DELAY + 1 , :]
    alpha57_vwap = self.vwap[di-DELAY: di-DELAY + 1 , :]

    close_argmax = np.zeros((2 , alpha57_close.shape[1]))
    close_argmax[0] = np.argmax(alpha57_closeadj[:-1 , ] , axis=0)
    close_argmax[1] = np.argmax(alpha57_closeadj[1: , ] , axis=0)

    alpha57 = 0 - (alpha57_close - alpha57_vwap) / self._decay_linear(self._rank(close_argmax))

    alpha = alpha57.reshape(alpha57_closeadj.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha 









def modified_alpha58():
    
    # ################################################################################
    # alpha58 = (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
    # alpha58 = (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 4), 8), 6))
    
    # 回测结果
    # Valify 测试通过 by xx
    # ################################################################################
    
    d1 = 4 
    d2 = 8 
    d3 = 6 

    alpha58_vwapadj = self.vwap[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:]
    alpha58_volume = self.Volume[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:]
    Industry = self.Industry[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:,:]

    alpha58_Ind_vwapadj = self._industry_neutral(alpha58_vwapadj, Industry)
    alpha58_Ind_volume = self._industry_neutral(alpha58_volume, Industry)
    alpha58_corr = np.zeros((d2+d3-1, alpha58_vwapadj.shape[1]))
    for ii in range(d2+d3-1):
        jj = ii + d1 
        alpha58_corr[ii] = self._correlation(alpha58_Ind_vwapadj[ii:jj], alpha58_Ind_volume[ii:jj])

    alpha58_decay = np.zeros((d3, alpha58_vwapadj.shape[1]))

    for ii in range(d3):
        jj = ii + d2 
        alpha58_decay[ii] = self._decay_linear(alpha58_corr[ii:jj])

    alpha58 = self._ts_rank(alpha58_decay)

    alpha = alpha58[0,:] * self.Universe_one.iloc[i,:]
    return alpha 







def modified_alpha59():
    
    
    # ################################################################################
    # alpha59 = (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
    # alpha59 = (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * w) + (vwap * (1 - w))), IndClass.industry), volume, 4), 16), 8))
    
    # 回测结果
    # Valify 测试通过 by xx
    # ################################################################################
    
    # rank: Ts_Rank(...)
    d1 = 4 
    d2 = 16 
    d3 = 8 

    alpha59_vwapadj = self.vwap[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:]
    alpha59_volume = self.Volume[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:]

    Industry = self.Industry[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:,:]


    alpha59_Ind_vwap = self._industry_neutral(alpha59_vwapadj, Industry)


    alpha59_corr = np.zeros((d2+d3-1, alpha59_vwapadj.shape[1]))
    for ii in range(d2+d3-1):
        jj = ii + d1
        alpha59_corr[ii] = self._correlation(alpha59_Ind_vwap[ii:jj], alpha59_volume[ii:jj])

    alpha59_decay = np.zeros((d3, alpha59_vwapadj.shape[1]))
    for ii in range(d3):
        jj = ii + d2
        alpha59_decay[ii] = self._decay_linear(alpha59_corr[ii:jj])

    alpha59_rank = self._ts_rank(alpha59_decay)

    alpha59 = -1 * alpha59_rank

    # 测试买卖反向
    # alpha59 = alpha59 * -1
    alpha = alpha59[0,:] * self.Universe_one.iloc[i,:]

    return alpha 







def modified_alpha60():

    # ################################################################################
    # alpha60 = (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(
    #     rank(ts_argmax(close , 10))))))
    
    # 回测结果
    # cum_return_rate           1.03262
    # final_return_rate        0.158352
    # beta                      1.05216
    # alpha                   0.0529116
    # sharpe ratio             0.302219
    # information ratio        0.179297
    # turnover rate          0.00377376
    # max drawdown             0.589944
    #   drawdown start date  2017-12-05
    #   drawdown end date          None
    # fitness                    1.9577
    # ################################################################################
    d = 10
    alpha60_closeadj = self.Closeprice[di-DELAY - d + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d + 1:di-DELAY + 1 , :]
    alpha60_close = self.Closeprice[di-DELAY:di-DELAY + 1 , :]
    alpha60_low = self.Lowprice[di-DELAY:di-DELAY + 1 , :]
    alpha60_volume = self.Volume[di-DELAY:di-DELAY + 1 , :]
    alpha60_high = self.Highprice[di-DELAY:di-DELAY + 1 , :]

    alpha60_1 = self._scale(self._rank(((alpha60_close - alpha60_low) - (alpha60_high - alpha60_close)) / (alpha60_high - alpha60_low) * alpha60_volume))

    alpha60_2 = self._scale(self._rank(np.argmax(alpha60_closeadj , axis=0))).reshape(1 , alpha60_close.shape[1])

    alpha60 = 0 - (2 * alpha60_1 - alpha60_2)

    alpha = alpha60.reshape(alpha60_closeadj.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha








def modified_alpha61():
    
    # ################################################################################
    # Alpha 61: (rank((vwap - ts_min(vwap, 16))) < rank(correlation(vwap, adv180, 18)))
    # 修改为 ts_min(vwap, 16)
    # valify 验证通过 by WJY
    # ################################################################################
    
    d1 = 16
    d0 = 180
    d2 = 18

    alpha61_vwapadj = self.vwap[di-DELAY-d2+1:di-DELAY + 1] * self.adjfactor[di-DELAY-d2 + 1:di-DELAY+1]
    alpha61_volume = self.Volume[di - DELAY - d1 - (d2 - 1) + 1:di-DELAY + 1 , :]

    alpha61_adv180 = np.zeros((d2, alpha61_vwapadj.shape[1]))
    for ii in range(d2):
        jj = ii + d0
        alpha61_adv180[ii] = np.nanmean(alpha61_volume[ii:jj], axis=0, keepdims=True)

    alpha61_rank1 = self._rank(alpha61_vwapadj[d1-1:d1,:] - self._ts_min(alpha61_vwapadj))
    alpha61_rank2 =  self._rank(self._correlation(alpha61_vwapadj, alpha61_adv180))

    alpha61 = alpha61_rank1 < alpha61_rank2 

    alpha = alpha61[0,:] * self.Universe_one.iloc[i,:]


    return alpha 


def modified_alpha62():
    # ################################################################################
    # Alpha62: ((rank(correlation(vwap , sum(adv20 , 22) , 10)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    
    # 回测结果
    #
    # cum_return_rate          0.728746
    # final_return_rate        0.120125
    # beta                      0.96868
    # alpha                   0.0202371
    # sharpe ratio             0.221977
    # information ratio       0.0608349
    # turnover rate            0.246722
    # max drawdown             0.580085
    #   drawdown start date  2017-12-25
    #   drawdown end date          None
    # fitness                   0.15489
    # ################################################################################    
    d1 = 20
    d2 = 22
    d3 = 10

    volume = self.Volume[di-DELAY - d1 - (d2 - 1) - (d3 - 1) + 1:di-DELAY + 1 , :]
    open = self.Openprice[di-DELAY: di-DELAY + 1 , :]
    high = self.Highprice[di-DELAY: di-DELAY + 1 , :]
    low = self.Lowprice[di-DELAY: di-DELAY + 1 , :]
    vwap = self.vwap[di-DELAY - d3 + 1: di-DELAY + 1 , :]

    adv20 = np.zeros(((d2 + d3 - 1 , volume.shape[1])))
    for ii in range(d2 + d3 - 1):
        jj = ii + d1
        adv20[ii] = np.nanmean(volume[ii:jj])

    sumadv20 = np.zeros(((d3) , volume.shape[1]))
    for ii in range(d3):
        jj = ii + d2
        sumadv20[ii] = np.nansum(adv20[ii:jj , :] , axis=0 , keepdims=True)

    rank1 = self._rank(self._correlation(vwap , sumadv20))
    rank2 = self._rank(self._rank(open) * 2 < (self._rank((high + low) / 2) + self._rank(high)))

    alpha62 = (rank1 < rank2) * -1

    # 买卖反向
    alpha62 = alpha62 * -1
    alpha = alpha62.reshape(open.shape[1] , ) * self.Universe_one.iloc[i , :]
    return alpha



def modified_alpha63():
    
    # ################################################################################
    # Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
    # Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2), 8)) 
    #              - 
    #             rank(decay_linear(correlation(((vwap * w) + (open * (1 - w))), sum(adv180, 37), 14), 12))) 
    #           * -1)
    
    
    # 回测结果
    # Valify 验证通过 by xx
    # ################################################################################
    # 注意 需要将 lookback_days 设为 252 或者更大
    w = 0.318108
    d1 = 2 
    d2 =8 

    d0 = 180 
    d3 = 37 
    d4 = 14 
    d5 = 12

    alpha63_closeadj = self.Closeprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] 
    Industry = self.Industry[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:,:]

    alpha63_volume = self.Volume[di-DELAY-(d0+d3-1+d4-1+d5-1)+1:di-DELAY+1,:]
    alpha63_vwapadj = self.vwap[di-DELAY-(d4+d5-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d4+d5-1)+1:di-DELAY+1,:] 
    alpha63_openadj = self.Openprice[di-DELAY-(d4+d5-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d4+d5-1)+1:di-DELAY+1,:] 

    # rank1 
    alpha63_rank1_Ind_close = self._industry_neutral(alpha63_closeadj, Industry)
    alpha63_rank1_delta = np.zeros((d2, alpha63_closeadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha63_rank1_delta[ii] = self._delta(alpha63_rank1_Ind_close[ii:jj])

    alpha63_rank1 = self._rank(self._decay_linear(alpha63_rank1_delta))


    # rank2 
    alpha63_rank2_adv180 = np.zeros((d3+d4-1+d5-1, alpha63_closeadj.shape[1]))
    for ii in range(d3+d4-1+d5-1):
        jj = ii + d0 
        alpha63_rank2_adv180[ii] = np.nanmean(alpha63_volume[ii:jj], axis=0, keepdims=True)

    alpha63_rank2_sum_adv180 = np.zeros((d4+d5-1, alpha63_closeadj.shape[1]))
    for ii in range(d4+d5-1):
        jj = ii + d3 
        alpha63_rank2_sum_adv180[ii] = np.nansum(alpha63_rank2_adv180[ii:jj], axis=0, keepdims=True)

    alpha63_rank2_corr = np.zeros((d5,alpha63_closeadj.shape[1]))
    for ii in range(d5):
        jj = ii + d4 
        alpha63_rank2_corr[ii] = self._correlation((alpha63_vwapadj * w + alpha63_openadj * (1-w))[ii:jj], alpha63_rank2_sum_adv180[ii:jj])

    alpha63_rank2 = self._rank(self._decay_linear(alpha63_rank2_corr))

    alpha63 = (alpha63_rank1 - alpha63_rank2) * -1
    # 测试买卖反向 
    alpha63 = alpha63 * -1 
    alpha = alpha63[0,:] * self.Universe_one.iloc[i,:]
    return alpha 









def modified_alpha64():
    # ################################################################################
    # Alpha64: ((rank(
    #     correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))) , 12.7054) , sum(adv120 , 12.7054) , 16.6208)) < rank(
    #     delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))) , 3.69741))) * -1)
    
    # 回测结果 买卖反向操作
    ##cum_return_rate           1.08473
    ##final_return_rate        0.164445
    ##beta                     0.954353
    ##alpha                   0.0655091
    ##sharpe ratio             0.343666
    ##information ratio         0.21281
    ##turnover rate            0.146709
    ##max drawdown              0.57762
        ##drawdown start date  2017-12-05
        ##drawdown end date          None
    ##fitness                  0.363846
    # 近一年下行   
    # ################################################################################    
    d1 = 13
    d2 = 120
    d3 = 17
    d4 = 4

    volume = self.Volume[di-DELAY - d2 - (d1 - 1) - (d3 - 1) + 1:di-DELAY + 1 , :]
    highadj = self.Highprice[di-DELAY - d1 - (d3 - 1) + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d1 - (d3 - 1) + 1:di-DELAY + 1 , :]
    vwapadj = self.vwap[di-DELAY - d1 - (d3 - 1) + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d1 - (d3 - 1) + 1:di-DELAY + 1 , :]
    lowadj = self.Lowprice[di-DELAY - d1 - (d3 - 1) + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d1 - (d3 - 1) + 1:di-DELAY + 1 , :]
    openadj = self.Openprice[di-DELAY - d1 - (d3 - 1) + 1:di-DELAY + 1 , :] * self.adjfactor[di-DELAY - d1 - (d3 - 1) + 1:di-DELAY + 1 , :]

    rank1_adv120 = np.zeros((d1 + d3 - 1 , volume.shape[1]))
    for ii in range(d1 + d3 - 1):
        jj = ii + d2
        rank1_adv120[ii] = np.nanmean(volume[ii:jj], axis = 0, keepdims = True)

    rank1_sumadv120 = np.zeros((d3 , volume.shape[1]))
    for ii in range(d3):
        jj = ii + d1
        rank1_sumadv120[ii] = np.nansum(rank1_adv120[ii:jj,:] , axis=0 , keepdims=True)

    # 首先提取 当日前 d1+d3-1天 的数据 然后计算 sum
    rank1_openadj = openadj[-(d1 + d3 - 1): , :]
    rank1_lowadj = lowadj[-(d1 + d3 - 1): , :]
    rank1_sum_open_low = np.zeros((d3 , openadj.shape[1]))
    for ii in range(d3):
        jj = ii + d1
        rank1_sum_open_low[ii] = np.nansum(rank1_openadj[ii:jj, :] * 0.178404 + rank1_lowadj[ii:jj,:] * (1 - 0.178404), axis = 0, keepdims = True)

    rank1 = self._rank(self._correlation(rank1_sum_open_low , rank1_sumadv120))

    # 首先提取 当日前 d4天 的数据 然后计算 sum
    rank2_high = highadj[-d4: , :]
    rank2_low = lowadj[-d4: , :]
    rank2_vwap = vwapadj[-d4: , :]
    rank2 = self._rank(self._delta((rank2_high + rank2_low) / 2 * 0.178404 + rank2_vwap * (1 - 0.178404)))

    alpha64 = (rank1 < rank2) * -1
    # 测试买卖反向 
    alpha64 = alpha64 * -1
    alpha = alpha64.reshape(openadj.shape[1] , ) * self.Universe_one.iloc[i , :]    

    return alpha






def modified_alpha65():
    
    # ################################################################################
    # Alpha65=((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    
    # Alpha65=((rank(correlation(((open * 0.w) + (vwap * (1 - w))), sum(adv60, 9), 6)) < rank((open - ts_min(open, 14)))) * -1)
    
    # 回测结果 # 买卖反向
    # 代码已经验证无误（2018-07-12 11:11:00)
    # cum_return_rate          0.526262
    # final_return_rate       0.0915779
    # beta                     0.242661
    # alpha                   0.0399698
    # sharpe ratio             0.277417
    # information ratio      -0.0387496
    # turnover rate           0.0348013
    # max drawdown             0.315495
    #   drawdown start date  2017-12-25
    #   drawdown end date          None
    # fitness                  0.450019
    # ################################################################################  
    
    w = 0.00817205
    d1 = 6
    d2 = 9
    d3 = 60
    d4 = 14

    openadj = self.Openprice[di-DELAY - d1 + 1:di-DELAY + 1 , :]  # open vwap 取 6.40374天
    vwapadj = self.vwap[di-DELAY - d1 + 1:di-DELAY + 1 , :]  # open vwap 取 6.40374天
    volume = self.Volume[di-DELAY - (d3 + d2 - 1 + d1 - 1) + 1:di-DELAY + 1 , :]

    openadj2 = self.Openprice[di-DELAY - d4 + 1:di-DELAY + 1 , :]

    rank1_adv60 = np.zeros((d2 + d1 - 1 , openadj.shape[1]))
    for ii in range(d2 + d1 - 1):
        jj = ii + d3
        rank1_adv60[ii] = np.nanmean(volume[ii:jj , :])

    rank1_sumadv60 = np.zeros((d1 , openadj.shape[1]))
    for ii in range(d1):
        jj = ii + d2
        rank1_sumadv60[ii] = np.nansum(rank1_adv60 , axis=0 , keepdims=True)

    rank1 = self._rank(self._correlation((openadj[-d1: , :] * w + vwapadj[-d1: , :] * (1 - w)) , rank1_sumadv60))

    rank2 = self._rank(openadj2[d4 - 1:d4 , :] - self._ts_min(openadj2))

    alpha65 = (rank1 < rank2) * -1

    # 买卖反向
    alpha65 = alpha65 * -1
    alpha = alpha65.reshape(openadj.shape[1] , ) * self.Universe_one.iloc[i , :]

    return alpha 




def modified_alpha66():
    # ################################################################################
    # Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    
    # Alpha66: ((rank(decay_linear(delta(vwap , 3) , 7)) + Ts_Rank(
    #  decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))) , 11) , 7)) * -1)
    # 回测结果
    #
    # 买卖反向 最近一年下行
    #
    # cum_return_rate           1.53523
    # final_return_rate        0.212627
    # beta                     0.909273
    # alpha                    0.116689
    # sharpe ratio             0.499796
    # information ratio         0.40032
    # turnover rate            0.025299
    # max drawdown             0.515177
    # drawdown start date  2017-12-05
    # drawdown end date          None
    # fitness                   1.44894
    
    # ################################################################################    
    d1 = 4  # delta(vwap, 3)
    d2 = 11
    d3 = 7
    w = 0.96633

    # 4+7-1天的vwap数据 用于计算 rank1
    vwap1adj = self.vwap[di-DELAY-(d1+d3-1)+1:di-DELAY+1] * self.adjfactor[di-DELAY-(d1+d3-1)+1:di-DELAY+1]

    lowadj = self.Lowprice[di-DELAY-(d2+d3-1)+1:di-DELAY+1 , :] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1 , :]
    highadj = self.Highprice[di-DELAY-(d2+d3-1)+1:di-DELAY+1 , :] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1 , :]
    vwap2adj = self.vwap[di-DELAY-(d2+d3-1)+1:di-DELAY+1 , :] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1 , :]
    openadj = self.Openprice[di-DELAY-(d2+d3-1)+1:di-DELAY+1 , :] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1 , :]

    delta_vwap1adj = np.zeros((d3 , vwap1adj.shape[1]))
    for ii in range(d3):
        jj = ii + d1
        delta_vwap1adj[ii] = self._delta(vwap1adj[ii:jj])
    rank1 = self._rank(self._decay_linear(delta_vwap1adj))

    rank2_price = (lowadj * w + lowadj * (1 - w) - vwap2adj) / (openadj - (highadj + lowadj) / 2)
    decay_rank2_price = np.zeros((d3 , vwap1adj.shape[1]))
    for ii in range(d3):
        jj = ii + d2
        decay_rank2_price = self._decay_linear(rank2_price[ii:jj , :])

    rank2 = self._ts_rank(decay_rank2_price)

    alpha66 = (rank1 + rank2) * -1

    # 买卖反向
    alpha66 = alpha66 * -1 
    alpha = alpha66.reshape(openadj.shape[1] , ) * self.Universe_one.iloc[i , :]


    return alpha 

# ################################################################################
# Alpha67= ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
# Alpha67= ((rank((high - ts_min(high, 2)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6))) * -1)


# 回测结果
# valify 验证通过 by xx 
# ################################################################################

def modified_alpha67():

    d1 = 2

    d0=20 
    d2 = 6

    alpha67_highadj = self.Highprice[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:]
    alpha67_vwapadj = self.vwap[di-DELAY-(d2)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2)+1:di-DELAY+1,:]
    Industry = self.Industry[di-DELAY-(d2)+1:di-DELAY+1,:,:]

    alpha67_volume = self.Volume[di-DELAY-(d0+d2-1)+1:di-DELAY+1,:] 


    # rank1: rank((high-ts_min...))

    alpha67_rank1 = self._rank((alpha67_highadj[-1,:] - np.nanmin(alpha67_highadj, axis=0)).reshape(1, alpha67_highadj.shape[1]))

    # rank2: rank(correlation(...))
    alpha67_adv20 = np.zeros((d2,alpha67_highadj.shape[1]))
    for ii in range(d2):
        jj = ii + d0 
        alpha67_adv20[ii] = np.nanmean(alpha67_volume, axis=0, keepdims=True)

    alpha67_rank2 = self._rank(self._correlation(self._industry_neutral(alpha67_vwapadj, Industry), self._industry_neutral(alpha67_adv20, Industry)))

    alpha67 = (alpha67_rank1 ** alpha67_rank2 ) * -1
    # 测试买卖反向 
    # alpha67 = alpha67 * -1 
    alpha = alpha67[0,:] * self.Universe_one.iloc[i,:]

    return alpha 






# ################################################################################
# Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
# Alpha68: ((Ts_Rank(correlation(rank(high) , rank(adv15) , 9) , 14) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))) , 1))) * -1)

# 回测结果 买卖反向之后
## cum_return_rate           1.93929
## final_return_rate        0.250366
## beta                     0.998295
## alpha                    0.148508
## sharpe ratio             0.555938
## information ratio        0.496124
## turnover rate            0.302058
## max drawdown             0.541173
##   drawdown start date  2015-07-08
##   drawdown end date          None
## fitness                  0.506138
# ################################################################################

def modified_alpha68():

    d1 = 9
    d2 = 14
    d3 = 2
    d4 = 15
    w = 0.518371

    high = self.Highprice[di-DELAY - (d1 + d2 - 1) + 1: di-DELAY + 1 , :]
    volume = self.Volume[di-DELAY - (d4 + d1 - 1 + d2 - 1):di-DELAY + 1 , :]
    closeadj = self.Closeprice[di-DELAY - d3 + 1:di-DELAY + 1 , :]
    lowadj = self.Lowprice[di-DELAY - d3 + 1:di-DELAY + 1 , :]

    adv15 = np.zeros((d1 + d2 - 1 , high.shape[1]))
    for ii in range(d1 + d2 - 1):
        jj = ii + d4
        adv15[ii] = np.nanmean(volume[ii:jj], axis=0, keepdims = True)

    rank1_corr = np.zeros((d2 , high.shape[1]))

    for ii in range(d2):
        jj = ii + d1
        rank1_corr[ii] = self._correlation(self._rank(high[ii:jj]) , self._rank(adv15[ii:jj]))
    rank2_price = closeadj * w + lowadj * (1 - w)

    alpha68 = (self._ts_rank(rank1_corr) < self._rank(self._delta(rank2_price))) * -1
    #测试买卖 反向 
    alpha68 = alpha68 * -1 
    alpha = alpha68.reshape(lowadj.shape[1] , ) * self.Universe_one.iloc[i , :]


    return alpha 

# ################################################################################
# Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)
# Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 3), 5))^Ts_Rank(correlation(((close * w) + (vwap * (1 - w))), adv20, 5), 9)) * -1)


# 回测结果
# 待验证code 
# ################################################################################


def modified_alpha69():

    w = 0.490655
    d1 = 3 
    d2 = 5 

    d0 = 20
    d3 = 5 
    d4 = 9

    alpha69_vwapadj = self.vwap[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]
    Industry = self.Industry[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]

    alpha69_volume = self.Volume[di-DELAY-(d0+d3-1+d4-1)+1:di-DELAY+1,:]
    alpha69_closeadj2 = self.Closeprice[di-DELAY-(d3+d4-1):di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3+d4-1):di-DELAY+1,:]
    alpha69_vwapadj2 = self.vwap[di-DELAY-(d3+d4-1)+1] * self.adjfactor[di-DELAY-(d3+d4-1)+1]


    # rank1: rank(...)

    alpha69_rank1_delta = np.zeros((d2, alpha69_vwapadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha69_rank1_delta[ii] = self._delta(self._industry_neutral(alpha69_vwapadj, Industry))

    alpha69_rank1 = self._rank(self._ts_max(alpha69_rank1_delta))


    # rank2: Ts_Rank(...)
    alpha69_rank2_adv20 = np.zeros((d3+d4-1,alpha69_vwapadj.shape[1]))
    for ii in range(d3+d4-1):
        jj = ii + d0 
        alpha69_rank2_adv20[ii] = np.nanmean(alpha69_volume[ii:jj], axis=0, keepdims=True)

    alpha69_rank2_corr = np.zeros((d4,alpha69_vwapadj.shape[1]))
    for ii in range(d4):
        jj = ii + d3 
        alpha69_rank2_corr[ii] = self._correlation(alpha69_closeadj2[ii:jj], alpha69_vwapadj2[ii:jj])

    alpha69_rank2 = self._ts_rank(alpha69_rank2_corr)

    alpha69 = (alpha69_rank1 ** alpha69_rank2) * -1

    alpha = alpha69[0,:] * self.Universe_one.iloc[i,:]
    return alpha 





# ################################################################################
# Alpha#70: ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
# Alpha70: ((rank(delta(vwap, 1))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 18), 18)) * -1)


# 回测结果
# valify 通过 by xx
# ################################################################################


def modified_alpha70():

    d0 = 50 
    d1 = 1+1 
    d2 = 18
    d3 = 18 

    alpha70_vwapadj = self.vwap[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:] 
    alpha70_closeadj = self.Closeprice[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:]
    Industry = self.Industry[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:,:]
    alpha70_volume = self.Volume[di-DELAY-(d0+d2-1+d3-1)+1:di-DELAY+1,:]

    # rank1: rank(...)
    alpha70_rank1 = self._rank(self._delta(alpha70_vwapadj))


    # rank2: Ts_Rank(...)
    alpha70_rank2_Ind_close = self._industry_neutral(alpha70_closeadj, Industry)

    alpha70_adv50 = np.zeros((d2+d3-1,alpha70_vwapadj.shape[1]))
    for ii in range(d2+d3-1):
        jj = ii + d0 
        alpha70_adv50[ii] = np.nanmean(alpha70_volume[ii:jj], axis=0, keepdims=True)

    alpha70_rank2_corr = np.zeros((d2,alpha70_vwapadj.shape[1]))
    for ii in range(d3):
        jj = ii + d2
        alpha70_rank2_corr[ii] = self._correlation(alpha70_rank2_Ind_close[ii:jj], alpha70_adv50[ii:jj])

    alpha70_rank2 = self._ts_rank(alpha70_rank2_corr)

    alpha70 = (alpha70_rank1**alpha70_rank2) * -1

    # 测试买卖反向 
    # alpha70 = alpha70 * -1 
    alpha = alpha70[0,:] * self.Universe_one.iloc[i,:]
    return alpha 








# ################################################################################
# Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))

# Alpha71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close , 3) , Ts_Rank(adv180 , 12) , 18) , 4) , 16) ,
#              Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap))) ^ 2) , 16) , 4))

# 含义，两个 ts_rank中取较大值

#  ts_rank1:
#  ts_rank2:
# 回测结果
# valify 通过 by WJY

# ################################################################################


def modified_alpha71():

    d0 = 180
    d1 = 3
    d2 = 12
    d4 = 18
    d5 = 4
    d6 = 16

    volume = self.Volume[di-DELAY-(d0+d2-1+d4-1+d5-1+d6-1)+1:di-DELAY+1, :]
    closeadj = self.Closeprice[di-DELAY-(d1+d4-1+d5-1+d6-1)+1:di-DELAY+1, :] * self.adjfactor[di-DELAY-(d1+d4-1+d5-1+d6-1)+1:di-DELAY+1, :] 

    low = self.Lowprice[di-DELAY - (d5 + d6 - 1) + 1:di-DELAY + 1 , :]
    open = self.Openprice[di-DELAY - (d5 + d6 - 1) + 1:di-DELAY + 1 , :]
    vwap = self.vwap[di-DELAY - (d5 + d6 - 1) + 1:di-DELAY + 1 , :]

    rank_close = np.zeros(((d4+d5-1+d6-1),closeadj.shape[1]))
    for ii in range(d4+d5-1+d6-1):
        jj = ii + d1
        rank_close[ii] = self._ts_rank(closeadj[ii:jj , :])

    adv180 = np.zeros(((d2+d4-1+d5-1+d6-1),closeadj.shape[1]))
    for ii in range(d2+d4-1+d5-1+d6-1):
        jj = ii + d0
        adv180[ii] = np.nanmean(volume[ii:jj , :], axis=0, keepdims=True)

    rank_adv180 = np.zeros(((d4 + d5 - 1 + d6 - 1),closeadj.shape[1]))
    for ii in range(d4 + d5 - 1 + d6 - 1):
        jj = ii + d2
        rank_adv180[ii] = self._ts_rank(adv180[ii:jj , :])

    corr_rank = np.zeros(((d5 + d6 - 1) , volume.shape[1]))
    for ii in range(d5 + d6 - 1):
        jj = ii + d4
        corr_rank[ii] = self._correlation(rank_close[ii:jj] , rank_adv180[ii:jj])

    decay_corr_rank = np.zeros((d6 , volume.shape[1]))
    for ii in range(d6):
        jj = ii + d5
        decay_corr_rank[ii] = self._decay_linear(corr_rank[ii:jj , :])

    rank1 = self._ts_rank(decay_corr_rank)

    rank2_decay = np.zeros((d5,closeadj.shape[1]))
    for ii in range(d5):
        jj = ii + d6
        rank2_decay[ii] = self._decay_linear((self._rank(low + open - 2 * vwap)[ii:jj]) ** 2)

    rank2 = self._ts_rank(rank2_decay)

    alpha71 = np.max(np.row_stack((rank1 , rank2)) , axis=0 , keepdims=True)

    alpha = alpha71.reshape(volume.shape[1] , ) * self.Universe_one.iloc[i , :] 


    return alpha


# ################################################################################
# Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
# Alpha72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 9), 10)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3), Ts_Rank(volume, 18), 7), 3)))
# 

# 回测结果
# valify 通过 by WJY

# ################################################################################
def modified_alpha72():


    d0 = 40 # 用于计算 adv40 
    d1 = 9 
    d2 = 10 
    d3 = 3 
    d4 = 18 
    d5 = 7 
    d6 = 3


    alpha72_volume1 = self.Volume[di-DELAY-(d0+(d1-1)+(d2-1))+1: di-DELAY+1,:] * self.adjfactor[di-DELAY-(d0+(d1-1)+(d2-1))+1: di-DELAY+1,:]
    alpha72_highadj = self.Highprice[di-DELAY-(d1+(d2-1))+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+(d2-1))+1:di-DELAY+1,:]
    alpha72_lowadj = self.Highprice[di-DELAY-(d1+(d2-1))+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+(d2-1))+1:di-DELAY+1,:]       
    alpha72_vwapadj = self.vwap[di-DELAY-(d3+d5-1+d6-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3+d5-1+d6-1)+1:di-DELAY+1,:]
    alpha72_volume2 = self.vwap[di-DELAY-(d4+d5-1+d6-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d4+d5-1+d6-1)+1:di-DELAY+1,:]

    # corr1 共有3重： adv40-过去9天取correlation-过去10天取移动平均

    corr1_adv40 = np.zeros((d1+d2-1, alpha72_highadj.shape[1]))

    # 计算 adv40
    for ii in range(d1+d2-1):
        jj = ii + d0
        corr1_adv40[ii] = np.nanmean(alpha72_volume1[ii:jj,:], axis=0, keepdims=True)
    # 进一步在过去9天内取correlation
    corr1 = np.zeros((d2, alpha72_highadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        corr1[ii] = self._correlation((((alpha72_highadj+alpha72_lowadj)/2)[ii:jj, :]), corr1_adv40[ii:jj, :])
    # 最后去 decay_linear 以及取 rank
    rank1 = self._rank(self._decay_linear(corr1))



    corr2_tsrank1 = np.zeros((d5+d6-1,alpha72_highadj.shape[1]))
    corr2_tsrank2 = np.zeros((d5+d6-1,alpha72_highadj.shape[1])) 

    for ii in range(d5+d6-1):
        jj1 = ii+d3
        jj2 = ii+d4 
        corr2_tsrank1[ii] = self._ts_rank(alpha72_vwapadj[ii:jj1,:])
        corr2_tsrank2[ii] = self._ts_rank(alpha72_volume2[ii:jj2,:])

    corr2 = np.zeros((d6,alpha72_highadj.shape[1]))
    for ii in range(d6):
        corr2[ii] = self._correlation(corr2_tsrank1[ii:jj, :], corr2_tsrank2[ii:jj, :])

    rank2 = self._rank(self._decay_linear(corr2))

    alpha72 = rank1 / rank2 

    alpha = alpha72.reshape(alpha72_highadj.shape[1] , ) * self.Universe_one.iloc[i , :]


    return alpha

# ################################################################################
# Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
# Alpha73: 
# (max(rank(decay_linear(delta(vwap, 5), 3)), Ts_Rank(decay_linear(((delta(((open * w) + (low * (1 - w))), 2) / ((open * w) + (low * (1 - w)))) * -1), 3), 17)) * -1)


# 回测结果
# valify 通过 by WJY
# ################################################################################
def modified_alpha73():


    w = 0.147155

    d1 = 5 
    d2 = 3 

    d3 = 2
    d4 = 3
    d5 = 17 


    alpha73_vwapadj = self.vwap[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]

    alpha73_openadj = self.Openprice[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:]
    alpha73_lowadj = self.Lowprice[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:]

    alpha71_rank1_delta_vwap = np.zeros((d2, alpha73_vwapadj.shape[1]))

    for ii in range(d2):
        jj = ii + d1 
        alpha71_rank1_delta_vwap[ii] = self._delta(alpha73_vwapadj[ii:jj,:])

    alpha73_rank1 = self._rank(self._decay_linear(alpha71_rank1_delta_vwap)) 

    # open low 加权平均的增长率
    alpha73_rank2_openlow_gr = np.zeros(((d4+d5-1),alpha73_vwapadj.shape[1]))

    for ii in range(d4+d5-1):
        jj = ii + d3 
        alpha73_rank2_openlow_gr[ii] = self._delta(alpha73_openadj[ii:jj] * w + alpha73_lowadj[ii:jj] * (1-w)) / (alpha73_openadj[jj-1:jj] * w + alpha73_lowadj[jj-1:jj] * (1-w))

    # alpha73_rank2 =  self._ts_rank()

    alpha73_rank2_decay = np.zeros((d5,alpha73_vwapadj.shape[1]))
    for ii in range(d5):
        jj = ii + d4 
        alpha73_rank2_decay[ii] = self._decay_linear(-1 * alpha73_rank2_openlow_gr[ii:jj,:])

    alpha73_rank2 = self._ts_rank(alpha73_rank2_decay)

    alpha73 = -1 * np.max(np.row_stack((alpha73_rank1, alpha73_rank2)), axis=0, keepdims=True)
    alpha = alpha73.reshape((alpha73_openadj.shape[1],)) * self.Universe_one.iloc[i , :]


    return alpha












# ################################################################################
# Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) 	* -1)
# Alpha74: ((rank(correlation(close, sum(adv30, 37), 15)) < rank(correlation(rank(((high * w) + (vwap * (1 - w)))), rank(volume), 11))) * -1)


# 回测结果
# valify 通过 by WJY
# ################################################################################

def modified_alpha74():

    w = 0.0261661

    d0 = 30 
    d1 = 37
    d2 = 15

    d3 = 11 

    # rank1 相关数据
    alpha74_volume1 = self.Volume[di-DELAY-(d0+d1-1+d2-1)+1:di-DELAY+1,:] 
    alpha74_closeadj = self.Closeprice[di-DELAY-(d2)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2)+1:di-DELAY+1,:] 
    # rank2 相关数据
    alpha74_volume2 = self.Volume[di-DELAY-(d3)+1:di-DELAY+1,:] 
    alpha74_vwapadj = self.vwap[di-DELAY-(d3)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3)+1:di-DELAY+1,:] 
    alpha74_highadj = self.Highprice[di-DELAY-(d3)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3)+1:di-DELAY+1,:] 

    #计算 rank1 
    alpha74_rank1_adv30 = np.zeros(((d1+d2-1),alpha74_volume2.shape[1]))
    for ii in range(d1+d2-1):
        jj = ii + d0 
        alpha74_rank1_adv30[ii] = np.nanmean(alpha74_volume1[ii:jj,:], axis=0, keepdims=True)

    alpha74_rank1_sumadv = np.zeros((d2, alpha74_volume2.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha74_rank1_sumadv[ii] = np.nansum(alpha74_rank1_adv30[ii:jj,:], axis=0, keepdims=True)

    alpha74_rank1 = self._rank(self._correlation(alpha74_closeadj, alpha74_rank1_sumadv))

    # 计算rank2
    alpha74_rank2 = self._rank(self._correlation((alpha74_highadj * w + alpha74_vwapadj * (1-w)), self._rank(alpha74_volume2)))

    # 计算alpha74
    alpha74 = (alpha74_rank1 < alpha74_rank2) * -1

    alpha = alpha74.reshape((alpha74_closeadj.shape[1] ,))* self.Universe_one.iloc[i , :]


    return alpha 



# ################################################################################
# Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
# Alpha75: (rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12)))

# 回测结果
## cum_return_rate          0.665142
## final_return_rate        0.111458
## beta                      1.03099
## alpha                  0.00742534
## sharpe ratio             0.189241
## information ratio       0.0304056
## turnover rate            0.187285
## max drawdown             0.623283
##   drawdown start date  2017-12-05
##   drawdown end date          None
## fitness                  0.145988

# 15年收益明显上升，17年下降
# ################################################################################

def modified_alpha75():


    d0 = 50 
    d1 = 4
    d2 = 12 

    # 计算rank1的数据
    alpha75_vwapadj = self.vwap[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
    alph75_volume1 = self.Volume[di-DELAY-(d1)+1:di-DELAY+1,:]

    # 计算rank2的数据
    alpha75_low = self.Lowprice[di-DELAY-(d2)+1:di-DELAY+1,:]
    alpha75_volume2 = self.Volume[di-DELAY-(d0+d2-1)+1:di-DELAY+1,:]

    # 计算rank1
    alpha75_rank1 = self._rank(self._correlation(alpha75_vwapadj, alph75_volume1))

    # 计算rank2 
    alpha75_rank2_adv50 = np.zeros((d2, alpha75_vwapadj.shape[1]))
    for ii in range(d2):
        jj = ii + d0 
        alpha75_rank2_adv50[ii] = np.nanmean(alpha75_volume2[ii:jj,:], axis=0, keepdims=True)

    alpha75_rank2 = self._rank(self._correlation(self._rank(alpha75_low), self._rank(alpha75_rank2_adv50)))

    alpha75 = np.where(alpha75_rank1<alpha75_rank2, 1, 0)

    alpha = alpha75.reshape((alpha75_vwapadj.shape[1],)) * self.Universe_one.iloc[i , :]


    return alpha 










# ################################################################################
# Alpha#76: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
# Alpha76: (max(rank(decay_linear(delta(vwap, 1), 12)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8), 20), 17), 19)) * -1)


# 回测结果
# 待验证code
# ################################################################################


def modified_alpha76():
    d1 = 1+1
    d2 = 12 

    d0 = 81 
    d3 = 8 
    d4 = 20 
    d5 = 17 
    d6 = 19 

    alpha76_vwapadj = self.vwap[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] 


    alpha76_volume = self.Volume[di-DELAY-(d0+d3-1+d4-1+d5-1)+1:di-DELAY+1]
    alpha76_lowadj = self.Lowprice[di-DELAY-(d3+d4-1+d5-1+d6-1)+1:di-DELAY+1] * self.adjfactor[di-DELAY-(d3+d4-1+d5-1+d6-1)+1:di-DELAY+1]
    Industry = self.Industry[di-DELAY-(d3+d4-1+d5-1+d6-1)+1:di-DELAY+1,:,:]



    alpha76_rank1_delta = np.zeros((d2,alpha76_vwapadj.shape[1])) 
    for ii in range(d2):
        jj = ii + d1 
        alpha76_rank1_delta[ii] = self._delta(alpha76_vwapadj[ii:jj])

    alpha76_rank1 = self._rank(self._decay_linear(alpha76_rank1_delta))


    alpha76_rank2_adv81 = np.zeros((d3+d4-1+d5-1+d6-1,alpha76_vwapadj.shape[1]))
    for ii in range(d3+d4-1+d5-1+d6-1):
        jj = ii + d3 
        alpha76_rank2_adv81[ii] = np.nanmean(alpha76_volume[ii:jj], axis=0, keepdims=True)

    alpha76_rank2_Ind_low = self._industry_neutral(alpha76_lowadj, Industry)

    alpha76_rank2_corr = np.zeros((d4+d5-1+d6-1, alpha76_vwapadj.shape[1]))
    for ii in range(d4+d5-1+d6-1):
        jj = ii + d3 
        alpha76_rank2_corr[ii] = self._correlation(alpha76_rank2_Ind_low[ii:jj], alpha76_rank2_adv81[ii:jj])

    alpha76_rank2_tsrank_tsrank = np.zeros((d5+d6-1, alpha76_vwapadj.shape[1]))
    for ii in range(d5+d6-1):
        jj = ii + d4
        alpha76_rank2_tsrank_tsrank[ii] = self._ts_rank(alpha76_rank2_corr[ii:jj])

    alpha76_rank2_decay = np.zeros((d6, alpha76_vwapadj.shape[1]))
    for ii in range(d6):
        jj = ii + d5
        alpha76_rank2_decay[ii] = self._decay_linear(alpha76_rank2_tsrank_tsrank[ii:jj])

    alpha76_rank2 = self._ts_rank(alpha76_rank2_decay)

    alpha76 = np.where(alpha76_rank1 > alpha76_rank2, alpha76_rank1, alpha76_rank2) * -1
    # alpha76 = np.max(np.row_stack((alpha76_rank1, alpha76_rank2), axis=0)) * -1
    # 测试买卖反向
    alpha76 = alpha76 * -1 

    alpha = alpha76[0,:] * self.Universe_one.iloc[i,:]

    return alpha 








# ################################################################################
# Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
# Alpha77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3), 6)))


# Valify 验证通过 by WJY
# ################################################################################
def modified_alpha77():
    d0 = 40
    d1 = 20
    d2 = 3 
    d3 = 6

    # 用于计算rank1的数据
    alpha77_highadj1 = self.Highprice[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:]
    alpha77_lowadj1 = self.Lowprice[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:]
    alpha77_vwapadj = self.vwap[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:]

    # 用于计算rank2的数据
    alpha77_volume = self.Volume[di-DELAY-(d0+d2-1+d3-1)+1:di-DELAY+1,:]
    alpha77_highadj2 = self.Highprice[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:]
    alpha77_lowadj2 = self.Lowprice[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:]


    # 计算rank1
    alpha77_rank1 = self._rank(self._decay_linear((alpha77_highadj1+alpha77_lowadj1)/2-alpha77_vwapadj))

    # 计算rank2 
    alpha77_rank2_adv40 = np.zeros((d2+d3-1, alpha77_highadj1.shape[1]))
    for ii in range(d2+d3-1):
        jj = ii + d0 
        alpha77_rank2_adv40[ii] = np.nanmean(alpha77_volume[ii:jj], axis=0, keepdims=True)

    alpha77_rank2_corr = np.zeros((d3, alpha77_highadj1.shape[1]))
    for ii in range(d3):
        jj = ii + d2 
        alpha77_rank2_corr[ii] = self._correlation(((alpha77_highadj2[ii:jj] + alpha77_lowadj2[ii:jj])/2) , alpha77_rank2_adv40[ii:jj])

    alpha77_rank2 = self._rank(self._decay_linear(alpha77_rank2_corr))

    alpha77 = np.min(np.row_stack((alpha77_rank1, alpha77_rank2)), axis=0, keepdims=True)

    alpha = alpha77[0,:] * self.Universe_one.iloc[i , :]
    return alpha



# ################################################################################
# Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
# Alpha78: (rank(correlation(sum(((low * w) + (vwap * (1 - w))), 20), sum(adv40, 20), 7))^rank(correlation(rank(vwap), rank(volume), 6)))



# 回测结果
## cum_return_rate           1.07402
## final_return_rate        0.163202
## beta                      1.00231
## alpha                   0.0610771
## sharpe ratio             0.327748
## information ratio        0.202479
## turnover rate           0.0674819
## max drawdown             0.590967
##   drawdown start date  2017-12-05
##   drawdown end date          None
## fitness                  0.509693

## 15年收益明显上升，17年下降
# ################################################################################
def modified_alpha78():

    d0 = 40
    d1 = 20 
    d2 = 7
    d3 = 6
    w = 0.352233

    alpha78_lowadj = self.Lowprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] 
    alpha78_vwapadj1 = self.vwap[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] 
    alpha78_volume1 = self.Volume[di-DELAY-(d0+d1-1+d2-1)+1:di-DELAY+1] 

    alpha78_vwapadj2 = self.vwap[di-DELAY-d3+1:di-DELAY+1] * self.adjfactor[di-DELAY-d3+1:di-DELAY+1]
    alpha78_volume2 = self.Volume[di-DELAY-d3+1:di-DELAY+1]

    # rank1 过去7天内 价 量 的相关性 
    alpha78_rank1_adv40 = np.zeros((d1+d2-1, alpha78_lowadj.shape[1]))
    for ii in range(d1+d2-1):
        jj = ii + d0
        alpha78_rank1_adv40[ii] = np.nanmean(alpha78_volume1, axis = 0, keepdims = True) 

    alpha78_rank1_sumadv40 = np.zeros((d2, alpha78_lowadj.shape[1])) 
    for ii in range(d2):
        jj = ii + d1
        alpha78_rank1_sumadv40[ii] = np.nanmean(alpha78_rank1_adv40, axis = 0, keepdims = True)

    alpha78_rank1_sumprice = np.zeros((d2,alpha78_lowadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha78_rank1_sumprice[ii] = np.nanmean((alpha78_lowadj*w + alpha78_vwapadj1*(1-w))[ii:jj,:], axis = 0, keepdims = True)

    alpha78_rank1 = self._rank(self._correlation(alpha78_rank1_sumadv40, alpha78_rank1_sumprice))
    # rank2 过去6天内 价 量 的相关性

    alpha78_rank2 = self._rank(self._correlation(alpha78_vwapadj2, alpha78_volume2))

    alpha78 = alpha78_rank1**alpha78_rank2 
    alpha = alpha78[0,:] * self.Universe_one.iloc[i , :]

    return alpha








# ################################################################################
# Alpha#79: (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))
# Alpha79: (rank(delta(IndNeutralize(((close * w) + (open * (1 - w))), IndClass.sector), 1)) < rank(correlation(Ts_Rank(vwap, 4), Ts_Rank(adv150, 9), 15)))


# 回测结果
# Valify 验证通过 by WJY
# ################################################################################



def modified_alpha79():


    w = 0.60733

    d1 = 2 

    d0 = 150 
    d2 = 4 
    d3 = 9 
    d4 = 15

    alpha79_closeadj = self.Closeprice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
    alpha79_openadj = self.Openprice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
    Industry = self.Industry[di-DELAY-(d1)+1:di-DELAY+1,:,:] 

    alpha79_vwapadj = self.vwap[di-DELAY-(d2+d4-1)+1:di-DELAY+1,:]
    alpha79_volume = self.Volume[di-DELAY-(d0+d3-1+d4-1)+1:di-DELAY+1,:]

    alpha79_rank1 = self._rank(self._delta(self._industry_neutral((alpha79_closeadj * w + alpha79_openadj * (1-w)), Industry)))

    alpha79_rank2_adv150 = np.zeros((d3+d4-1,alpha79_closeadj.shape[1]))
    for ii in range(d3+d4-1):
        jj = ii + d0 
        alpha79_rank2_adv150[ii] = np.nanmean(alpha79_volume[ii:jj], axis=0, keepdims=True)

    alpha79_rank2_tsrank1 = np.zeros((d4,alpha79_closeadj.shape[1]))
    alpha79_rank2_tsrank2 = np.zeros((d4,alpha79_closeadj.shape[1]))

    for ii in range(d4):
        jj1 = ii + d2
        jj2 = ii + d3 
    alpha79_rank2_tsrank1[ii] = self._ts_rank(alpha79_vwapadj[ii:jj1])
    alpha79_rank2_tsrank2[ii] = self._ts_rank(alpha79_rank2_adv150[ii:jj2])

    alpha79_rank2 = self._rank(self._correlation(alpha79_rank2_tsrank1, alpha79_rank2_tsrank2))

    alpha79 = alpha79_rank1 < alpha79_rank2 

    alpha = alpha79[0,:] * self.Universe_one.iloc[i,:]

    return alpha 





# ################################################################################
# Alpha#80: ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
# Alpha80: ((rank(Sign(delta(IndNeutralize(((open * w) + (high * (1 - w))), IndClass.industry), 4)))^Ts_Rank(correlation(high, adv10, 5), 6)) * -1)


# 回测结果
# Valify 验证通过 by WJY
# ################################################################################


def modified_alpha80():
    w = 0.634196

    d0 = 10 

    d1 = 4 

    d2 = 5 
    d3 = 6 

    alpha80_openadj = self.Openprice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
    alpha80_highadj = self.Highprice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
    Industry = self.Industry[di-DELAY-(d1)+1:di-DELAY+1,:,:]
    alpha80_volume = self.Volume[di-DELAY-(d0+d2-1+d3-1)+1:di-DELAY+1]
    alpha80_highadj2 =self.Highprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]

    # rank1 : 
    alpha80_Ind_price = self._industry_neutral((alpha80_openadj * w + alpha80_highadj * (1-w) ), Industry)

    alpha80_rank1 = self._rank(np.sign(self._delta(alpha80_Ind_price)))


    # rank2 

    alpha80_rank2_adv10 = np.zeros((d2+d3-1, alpha80_openadj.shape[1]))
    for ii in range(d2+d3-1):
        jj = ii + d0 
        alpha80_rank2_adv10[ii] = np.nanmean(alpha80_volume[ii:jj], axis=0, keepdims=True)

    alpha80_rank2_corr = np.zeros((d3, alpha80_openadj.shape[1]))
    for ii in range(d3):
        jj = ii + d2 
        alpha80_rank2_corr[ii] = self._correlation(alpha80_volume[ii:jj], alpha80_rank2_adv10[ii:jj])

    alpha80_rank2 = self._ts_rank(alpha80_rank2_corr)	
    # Alpha80: ((rank(Sign(delta(IndNeutralize(((open * w) + (high * (1 - w))), IndClass.industry), 4)))^Ts_Rank(correlation(high, adv10, 5), 6)) * -1)

    alpha80 = alpha80_rank1**alpha80_rank2 

    alpha = alpha80[0,:] * self.Universe_one.iloc[i,:]
    return alpha




# ################################################################################
# Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
#Alpha81: ( ( rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 50), 8))^4)), 15))) 
#             < 
#             rank(correlation(rank(vwap), rank(volume), 5))
#           ) 
#           * -1
#         )


# 回测结果
# Valify 验证通过 by WJY
# ################################################################################



def modified_alpha81():


    from math import log
    d0 = 10 
    d1 = 50 
    d2 = 8 
    d3 = 15 
    d4 = 5 
    alha81_vwapadj = self.vwap[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:]
    alpha81_volume = self.Volume[di-DELAY-(d0+d1-1+d2-1+d3-1)+1:di-DELAY+1,:]
    alpha81_vwap = self.vwap[di-DELAY-(d4)+1:di-DELAY+1,:] 
    # rank1: rank(Log...) 

    alpha81_rank1_adv10 = np.zeros((d1+d2-1+d3-1, alha81_vwapadj.shape[1]))
    for ii in range(d1-1+d2-1+d3-1):
        jj = ii + d0 
        alpha81_rank1_adv10[ii] = np.nanmean(alpha81_volume[ii:jj], axis=0, keepdims=True)

    alpha81_rank1_sumadv10 = np.zeros((d2+d3-1, alha81_vwapadj.shape[1])) 
    for ii in range(d2+d3-1):
        jj = ii + d1
        alpha81_rank1_sumadv10[ii] = np.nansum(alpha81_rank1_adv10[ii:jj], axis=0, keepdims=True)

    alpha81_rank1_corr = np.zeros((d3, alha81_vwapadj.shape[1]))
    for ii in range(d3):
        jj = ii + d2
        alpha81_rank1_corr[ii] = self._correlation(alha81_vwapadj[ii:jj], alpha81_rank1_sumadv10[ii:jj])

    alpha81_rank1 = self._rank(log(np.nanprod(self._rank(self._rank(alpha81_rank1_corr)), axis=0, keepdims=True)))

    # rank2: rank(correlation...)
    alpha81_rank2 = self._rank(self._correlation(self._rank(alpha81_vwap), self._rank(alpha81_volume[-d4:])))

    alpha81 = (alpha81_rank1 < alpha81_rank2 ) * -1

    alpha = alpha81[0,:] * self.Universe_one.iloc[i,:]

    return alpha 






# ################################################################################
# Alpha#82: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +  (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
#Alpha82: ( min(
#               rank(decay_linear(delta(open, 1), 15)), 
#               Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * w) +  (close * (1 - w))), 17), 7), 13)
#              ) 
#         * -1
#         )


# 回测结果
# Valify 验证通过 by WJY
# ################################################################################

def modified_alpha82():


    d1 = 1 
    d2 = 15 

    d3 = 17
    d4 = 7 
    d5 = 13

    alpha82_openadj = self.Openprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfacto[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]

    alpha82_openadj2 = self.Openprice[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:] 
    alpha82_closeadj2 = self.Closeprice[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:] 
    alpha82_volume = self.Volume[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:]
    Industry = self.Industry[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:,:]

    # rank1: rank(decay....)
    alpha82_rank1 = self._rank(self._decay_linear(np.diff(alpha82_openadj)))

    # rank2: Ts_Rank(...)
    alpha82_rank2_corr = np.zeros((d4+d5-1, alpha82_openadj.shape[1]))
    for ii in range(d4+d5-1):
        jj = ii + d3 
        alpha82_rank2_corr[ii] = self._correlation(self._industry_neutral(alpha82_volume, Industry)[ii:jj], (alpha82_openadj2 * w + alpha82_closeadj2 * (1-w))[ii:jj] )

    alpha82_rank2_decay = np.zeros((d5, alpha82_openadj.shape[1]))
    for ii in range(d5):
        jj = ii + d4
        alpha82_rank2_decay[ii] = self._decay_linear(alpha82_rank2_corr[ii:jj])

    alpha82_rank2 = self._ts_rank(alpha82_rank2_decay)


    alpha82 = np.min(np.row_stack((alpha82_rank1,alpha82_rank2), axis=0))  # 已经是一个array的形式

    alpha = alpha82 * self.Universe_one.iloc[i,:]

    return alpha 









# ################################################################################
# Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
# Alpha83: ( (rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) 
#              / 
#            (((high - low) / (sum(close, 5) / 5)) / (vwap - close))
#          )


# Valify 验证通过 by WJY
# ################################################################################


def modified_alpha83():
    d1 = 5 
    d2 = 2+1 

    alpha83_closeadj = self.Closeprice[di-DELAY-d2+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d2+1:di-DELAY+1,:] 
    alpha83_highadj = self.Highprice[di-DELAY-d2+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d2+1:di-DELAY+1,:] 
    alpha83_lowadj = self.Lowprice[di-DELAY-d2+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d2+1:di-DELAY+1,:] 
    alpha83_volume = self.Volume[di-DELAY:di-DELAY+1,:]
    alpha83_vwap = self.vwap[di-DELAY:di-DELAY+1,:]
    alpha83_close = self.Closeprice[di-DELAY:di-DELAY+1,:] 

    alpha83_rank1_avg_close = np.zeros((d2,alpha83_closeadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha83_rank1_avg_close[ii] = np.nanmean(alpha83_closeadj[ii:jj], axis=0, keepdims=True)

    alpha83_rank1 = self._rank(((alpha83_highadj - alpha83_lowadj)/alpha83_rank1_avg_close)[0:1,:] * self._rank(self._rank(alpha83_volume)))

    alpha83_price = ((alpha83_highadj - alpha83_lowadj)/alpha83_rank1_avg_close)[2:3,:]/(alpha83_vwap- alpha83_close)

    alpha83 = alpha83_rank1/ alpha83_price 


    alpha = alpha83[0,:] * self.Universe_one.iloc[i,:]
    return alpha 









# ################################################################################
# Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
# Alpha84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15)), 20), delta(close, 5))


# Valify 验证通过 by WJY
# ################################################################################



def modified_alpha84():

    d1 = 15 
    d2 = 20 
    d3 = 5 

    alpha84_vwapadj = self.vwap[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] 
    alpha84_closeadj = self.Closeprice[di-DELAY-(d3)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3)+1:di-DELAY+1,:] 

    alpha84_tsrank_price = np.zeros((d2, alpha84_vwapadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha84_tsrank_price[ii] = self._ts_max(alpha84_vwapadj[ii:jj])

    alpha84_tsrank = self._ts_rank(alpha84_vwapadj[-d2:,:] - alpha84_tsrank_price)

    alpha84_delta = self._delta(alpha84_closeadj)

    alpha84 = alpha84_tsrank ** alpha84_delta

    alpha = alpha84[0,:] * self.Universe_one.iloc[i,:]

    return alpha












# ################################################################################
# Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
# Alpha85: (rank(correlation(((high * w) + (close * (1 - w))), adv30, 10))^rank(correlation(Ts_Rank(((high + low) / 2), 4), Ts_Rank(volume, 10), 7)))

# 回测结果
# 代码 valify 通过 by WJY
# ################################################################################
def modified_alpha85():

    d0 = 30 
    d1 = 10
    d2 = 2 
    d3 = 4 
    d4 = 7
    w = 0.876703

    alpha85_highadj1 = self.Highprice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
    alpha85_closeadj1 = self.Closeprice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
    alpha85_volume1 = self.Volume[di-DELAY-(d0+d1-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d0+d1-1)+1:di-DELAY+1,:]

    alpha85_voluem2 = alpha85_volume1[-(d1+d4-1):,:]
    alpha85_highadj2 = self.Highprice[di-DELAY-(d3+d4-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3+d4-1)+1:di-DELAY+1,:]
    alpha85_lowadj2 = self.Lowprice[di-DELAY-(d3+d4-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3+d4-1)+1:di-DELAY+1,:]
    # 计算 rank1， 由内向外 逐层计算
    alpha85_rank1_adv30 = np.zeros((d1,alpha85_highadj1.shape[1]))
    for ii in range(d1):
        jj = ii + d0 
        alpha85_rank1_adv30[ii] = np.nanmean(alpha85_volume1[ii:jj,:], axis = 0 , keepdims=True)

    alpha85_rank1 = self._rank(self._correlation(alpha85_highadj1 * w + alpha85_closeadj1 * (1-w), alpha85_rank1_adv30))

    # 计算 rank2， 由内向外 逐层计算
    alpha85_rank2_tsrank1 = np.zeros((d4, alpha85_highadj1.shape[1]))
    for ii in range(d4):
        jj = ii + d3 
        alpha85_rank2_tsrank1[ii] = self._ts_rank(((alpha85_highadj2 + alpha85_lowadj2)/2)[ii:jj,:])

    alpha85_rank2_tsrank2 = np.zeros((d4, alpha85_highadj1.shape[1]))
    for ii in range(d4):
        jj = ii + d1 
        alpha85_rank2_tsrank2[ii] = self._ts_rank(alpha85_voluem2[ii:jj,:])

    alpha85_rank2 = self._rank(self._correlation(alpha85_rank2_tsrank1, alpha85_rank2_tsrank2))

    alpha85 = alpha85_rank1**alpha85_rank2
    alpha = alpha85[0,:] * self.Universe_one.iloc[i,:]


    return alpha 









# ################################################################################
# Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
# Alpha86: ((Ts_Rank(correlation(close, sum(adv20, 15), 6), 20) < rank(((open + close) - (vwap + open)))) * -1)


# 回测结果
# 代码 valify 通过 by WJY
# ################################################################################
def modified_alpha86():
    d0 = 20 
    d1 = 15
    d2 = 6
    d3 = 20 

    # 用于计算ts_rank的函数
    alpha86_closeadj = self.Closeprice[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:] 
    alpha86_volume = self.Volume[di-DELAY-(d0+d1-1+d2-1+d3-1)+1:di-DELAY+1,:]

    alpha86_close = self.Closeprice[di-DELAY:di-DELAY+1,:]
    alpha86_vwap = self.vwap[di-DELAY:di-DELAY+1,:]

    alpha86_adv20 =	np.zeros((d1+d2-1+d3-1, alpha86_closeadj.shape[1]))
    alpha86_sumadv20 = np.zeros((d2+d3-1, alpha86_closeadj.shape[1]))

    for ii in range(d1+d2-1+d3-1):
        jj = ii + d0
        alpha86_adv20[ii] = np.nanmean(alpha86_volume[ii:jj,:], axis=0, keepdims=True)

    for ii in range(d2+d3-1):
        jj = ii + d2 
        alpha86_sumadv20[ii] = np.nansum(alpha86_adv20[ii:jj,:], axis=0, keepdims=True)

    alpha86_tsrank_corr = np.zeros((d3, alpha86_closeadj.shape[1]))
    for ii in range(d3):
        jj = ii + d2 
        alpha86_tsrank_corr[ii] = self._correlation(alpha86_closeadj[ii:jj,:], alpha86_sumadv20[ii:jj,:])

    alpha86_tsrank = self._ts_rank(alpha86_tsrank_corr)
    alpha86_rank = self._rank(alpha86_close - alpha86_vwap)

    alpha86 = (alpha86_tsrank < alpha86_rank) * -1


    alpha = alpha86[0,:] * self.Universe_one.iloc[i,:]
    return alpha 








# ################################################################################
# Alpha#87: (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
# Alpha87: (max(
#                rank(decay_linear(delta(((close * w) + (vwap * (1 - w))), 2), 3)), 
#                Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13)), 5), 14)
#              ) 
#           * -1
#          )


# 回测结果
# Valify 验证通过 by WJY
# ################################################################################
def modified_alpha87():
    # 注意 look_back_days 设定为 150 
    # 注意 look_back_days 设定为 150 
    w = 0.369701
    d0 = 81
    d1 = 2 
    d2 = 3 

    d3 = 13 
    d4 = 5 
    d5 = 14

    alpha87_closeadj = self.Closeprice[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:]
    alpha87_vwapadj = self.vwap[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:]
    alpha87_closeadj2 = self.Closeprice[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:]
    alpha87_volume = self.Volume[di-DELAY-(d0+d3-1+d4-1+d5-1)+1:di-DELAY+1,:]
    Industry = self.Industry[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:,: ]

    # rank1: rank(....)
    alpha87_rank1_delta = np.zeros((d2, alpha87_closeadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha87_rank1_delta[ii] = self._delta((alpha87_closeadj * w + alpha87_vwapadj * (1-w))[ii:jj])

    alpha87_rank1 = self._rank(self._decay_linear(alpha87_rank1_delta))

    # rank2: Ts_Rank(...)

    alpha87_rank2_adv81 = np.zeros((d3+d4-1+d5-1,alpha87_closeadj.shape[1]))
    for ii in range(d3+d4-1+d5-1):
        jj = ii + d0 
        alpha87_rank2_adv81[ii] = np.nanmean(alpha87_volume[ii:jj], axis=0, keepdims=True)

    alpha87_rank2_Ind_adv81	= self._industry_neutral(alpha87_rank2_adv81, Industry)

    alpha87_rank2_corr = np.zeros((d4+d5-1, alpha87_closeadj.shape[1]))
    for ii in range(d4+d5-1):
        jj = ii + d3 
        alpha87_rank2_corr[ii] = self._correlation(alpha87_rank2_Ind_adv81[ii:jj], alpha87_closeadj2[ii:jj])

    alpha87_rank2_decay = np.zeros((d5, alpha87_closeadj.shape[1]))
    for ii in range(d5):
        jj = ii + d4 
        alpha87_rank2_decay[ii] = self._decay_linear(alpha87_rank2_corr[ii:jj])


    alpha87_rank2 = self._ts_rank(alpha87_rank2_decay)

    alpha87 = np.where(alpha87_rank1 > alpha87_rank2, alpha87_rank1, alpha87_rank2)*-1
    # 测试买卖反向 
    # alpha87 = alpha87 * -1 
    alpha = alpha87[0,:] * self.Universe_one.iloc[i,:]

    return alpha 





# ################################################################################
# Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
# Alpha88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8), Ts_Rank(adv60, 20), 8), 6), 3))
#
#
# 买入策略：两个排名取小作为alpha，横截面排序X 以及 时序排序Y 均较大的买入
# X：（open排名-close排名 low排名-high排名 ）在过去8天的时间加权平均，open越高close越低 & low越高high越低，加权平均越小，排名越大，考虑买入
# X：买入近期日收益率低，价格范围小的股票
# Y：近3天的时序排序：过去6天 Y1与Y2时序相关系数的加权平均
# Y1：收盘价8天时序排序  Y2：60日平均交易量8天时序排序
# Y：近期开始的平均交易量与收盘价相关性减弱 --> 相关系数加权平均小 --> 时序排序Y大 --> 考虑买入
# 买入：近期开始出现：日收益率低、价格值域小  同时  交易量与收盘价相关性减弱

# 回测结果
## 
## cum_return_rate           2.37279
## final_return_rate        0.286527
## beta                     0.797928
## alpha                    0.197994
## sharpe ratio             0.829806
## information ratio        0.781916
## turnover rate           0.0661096
## max drawdown             0.400249
##   drawdown start date  2015-07-08
##   drawdown end date          None
## fitness                   1.72754

## 15年收益明显上升，17年收益下降

# ################################################################################
def modified_alpha88():

    d0 = 60
    d1 = 8
    d2 = 20
    d3 = 6
    d4 = 3

    alpha88_open = self.Openprice[di-DELAY-(d1)+1:di-DELAY+1,:] 
    alpha88_low =  self.Lowprice[di-DELAY-(d1)+1:di-DELAY+1,:] 
    alpha88_high = self.Highprice[di-DELAY-(d1)+1:di-DELAY+1,:]
    alpha88_close = self.Closeprice[di-DELAY-(d1)+1:di-DELAY+1,:]

    alpha88_volume = self.Volume[di-DELAY+(d0+d2-1+d3-1+d1-1+d4-1)-1:di-DELAY+1,:]
    alpha88_closeadj2 = self.Closeprice[di-DELAY-(d1+d3-1+d4-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d3-1+d4-1)+1:di-DELAY+1,:]

    alpha88_rank1 = self._rank(self._decay_linear(self._rank(alpha88_open) + self._rank(alpha88_low)-self._rank(alpha88_high) - self._rank(alpha88_close)))



    alpha88_rank2_tsrank1 = np.zeros((d3+d4-1, alpha88_open.shape[1]))
    for ii in range(d3+d4-1):
        jj = ii + d1
        alpha88_rank2_tsrank1[ii] = self._ts_rank(alpha88_closeadj2[ii:jj,:])

    alpha88_rank2_adv60 = np.zeros((d2+d3-1+d1-1+d4-1, alpha88_open.shape[1]))
    for ii in range(d2+d3-1+d1-1+d4-1):
        jj = ii + d0 
        alpha88_rank2_adv60[ii] = np.nanmean(alpha88_volume[ii:jj], axis=0, keepdims=True)

    alpha88_rank2_tsrank2_adv60 = np.zeros((d1+d3-1+d4-1, alpha88_open.shape[1]))
    for ii in range(d1+d3-1+d4-1):
        jj = ii + d2 
        alpha88_rank2_tsrank2_adv60[ii] = self._ts_rank(alpha88_rank2_adv60[ii:jj,:])

    alpha88_rank2_tsrank2 = np.zeros((d3+d4-1, alpha88_open.shape[1]))
    for ii in range(d3+d4-1):
        jj = ii + d1 
        alpha88_rank2_tsrank2[ii] = self._ts_rank(alpha88_rank2_tsrank2_adv60[ii:jj,:])

    alpha88_rank2_corr = np.zeros((d4, alpha88_open.shape[1]))
    for ii in range(d4):
        jj = ii + d3 
        alpha88_rank2_corr[ii] = self._correlation(alpha88_rank2_tsrank1[ii:jj,:], alpha88_rank2_tsrank2[ii:jj,:])

    alpha88_rank2 = self._rank(self._decay_linear(alpha88_rank2_corr))


    alpha88 = np.where(alpha88_rank1 < alpha88_rank2, alpha88_rank1, alpha88_rank2)
    alpha = alpha88[0,:] * self.Universe_one.iloc[i,:]



    return alpha




# ################################################################################
# Alpha#89: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
# Alpha89: (  Ts_Rank(decay_linear(correlation(((low * w) + (high * (1 - w))), adv10, 7), 6), 4) 
#           - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3), 10), 15)  
#          )


# 回测结果
# Valify 验证通过  by WJY
# ################################################################################
def modified_alpha89():

    w = 0.967285
    d0 = 10 
    d1 = 7 
    d2 = 6 
    d3 = 4 
    alpha89_lowadj = user.Lowprice[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:] * user.adjfactor[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:]
    alpha89_highadj = user.Highprice[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:] * user.adjfactor[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:]
    alpha89_volume = user.Volume[di-DELAY-(d0+d1-1+d2-1+d3-1)+1:di-DELAY+1,:]


    # rank1: Ts_Rank(decay_linear(correlation...))
    alpha89_rank1_adv10 = np.zeros((d1+d2-1+d3-1, alpha89_lowadj.shape[1]))
    for ii in range(d1+d2-1+d3-1):
        jj = ii + d0 
        alpha89_rank1_adv10[ii] = np.nanmean(alpha89_volume, axis=0, keepdims=True)

    alpha89_rank1_corr = np.zeros((d2+d3-1,alpha89_lowadj.shape[1]))
    for ii in range(d2+d3-1):
        jj = ii + d1 
        alpha89_rank1_corr[ii] = user._correlation((alpha89_lowadj * w + alpha89_highadj * (1-w))[ii:jj], alpha89_rank1_adv10[ii:jj] ) 

    alpha89_rank1_decay = np.zeros((d3,alpha89_lowadj.shape[1]))
    for ii in range(d3):
        jj = ii + d2 
        alpha89_rank1_decay[ii] = user._decay_linear(alpha89_rank1_corr[ii:jj])

    alpha89_rank1 = user._ts_rank(alpha89_rank1_decay)


    # rank2: Ts_Rank(decay_linear(delta...))

    d4 = 3 
    d5 = 10 
    d6 = 15 

    alpha89_vwapadj = user.vwap[di-DELAY-(d4+d5-1+d6-1)+1:di-DELAY+1,:] * user.adjfactor[di-DELAY-(d4+d5-1+d6-1)+1:di-DELAY+1,:]
    alpha89_industry = user.Industry[di-DELAY-(d4+d5-1+d6-1)+1:di-DELAY+1,:]
    alpha89_Ind_vwapadj = user._industry_neutral(alpha89_vwapadj, alpha89_industry)

    alpha89_rank2_delta = np.zeros((d5+d6-1, alpha89_lowadj.shape[1]))
    for ii in range(d5+d6-1):
        jj = ii + d4 
        alpha89_rank2_delta[ii] = user._delta(alpha89_Ind_vwapadj[ii:jj])

    alpha89_rank2_decay = np.zeros((d6, alpha89_lowadj.shape[1]))
    for ii in range(d6):
        jj = ii + d5 
        alpha89_rank2_decay[ii] = user._decay_linear(alpha89_rank2_delta[ii:jj])

    alpha89_rank2 = user._ts_rank(alpha89_rank2_decay)
    alpha89 = alpha89_rank1 - alpha89_rank2 
    alpha = alpha89[0,:] * user.Universe_one.iloc[i,:]


    return alpha




# ################################################################################
# Alpha#90: ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
# Alpha#90: ((rank((close - ts_max(close, 5)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5), 3)) * -1)


# 回测结果
# Valify 验证通过 by WJY
# ################################################################################
def modified_alpha90():

    d0 = 40 
    d1 = 5 
    d2 = 3 

    alpha90_close = self.Closeprice[di-DELAY:di-DELAY+1] 
    alpha90_closeadj = self.Closeprice[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:] 
    alpha90_volume = self.Volume[di-DELAY-(d0+d1-1+d2-1)+1:di-DELAY+1,:] 
    alpha90_lowadj = self.Lowprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]
    Industry = self.Industry[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:,:]


    # rank1: rank(...)
    alpha90_rank1 = self._rank(alpha90_close - self._ts_max(alpha90_closeadj))

    # rank2: Ts_Rank(...)

    alpha90_rank2_adv40 = np.zeros((d1+d2-1, alpha90_close.shape[1]))
    for ii in range(d1+d2-1):
        jj = ii + d0 
        alpha90_rank2_adv40[ii] = np.nanmean(alpha90_volume, axis=0,keepdims=True)

    alpha90_rank2_Ind_adv40 = self._industry_neutral(alpha90_rank2_adv40, Industry)

    alpha90_rank2_corr = np.zeros((d2, alpha90_close.shape[1]))
    for ii in range(d2):
        jj = ii + d1
        alpha90_rank2_corr[ii] = self._correlation(alpha90_lowadj[ii:jj], alpha90_rank2_Ind_adv40[ii:jj])

    alpha90_rank2 = self._ts_rank(alpha90_rank2_corr)
    alpha90 = (alpha90_rank1 ** alpha90_rank2 ) * -1
    alpha = alpha90[0,:] * self.Universe_one.iloc[i,:]

    return alpha



# ################################################################################
# Alpha#91: ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
# Alpha#91  ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 10), 16), 4), 5) - rank(decay_linear(correlation(vwap, adv30, 4), 3)) * -1)


# 回测结果
# code valify 通过  by xx
# ################################################################################
def modified_alpha91():

    d0 = 30 
    d1 = 10 
    d2 = 16 
    d3 = 4 
    d4 = 5 

    d5 = 4 
    d6 = 3

    alpha91_closeadj = self.Closeprice[di-DELAY-(d1+d2-1+d3-1+d4-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1+d3-1+d4-1)+1:di-DELAY+1,:]
    alpha91_volume1 = self.Volume[di-DELAY-(d1+d2-1+d3-1+d4-1)+1:di-DELAY+1,:] 

    alpha91_vwapadj = self.vwap[di-DELAY-(d5+d6-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d5+d6-1)+1:di-DELAY+1,:]
    alpha91_volume2 = self.Volume[di-DELAY-(d0+d5-1+d6-1)+1:di-DELAY+1,:]

    Industry = self.Industry[di-DELAY-(d1+d2-1+d3-1+d4-1)+1:di-DELAY+1,:,: ]

    alpha91_Ind_close = self._industry_neutral(alpha91_closeadj, Industry)


    alpha91_rank1_corr = np.zeros((d2+d3-1+d4-1, alpha91_closeadj.shape[1]))
    for ii in range(d2+d3-1+d4-1):
        jj = ii + d1 
        alpha91_rank1_corr[ii] = self._correlation(alpha91_Ind_close[ii:jj], alpha91_volume1[ii:jj])

    alpha91_rank1_decay_decay = np.zeros((d3+d4-1, alpha91_closeadj.shape[1]))
    for ii in range(d3+d4-1):
        jj = ii + d2 
        alpha91_rank1_decay_decay[ii] = self._decay_linear(alpha91_rank1_corr[ii:jj])

    alpha91_rank1_decay = np.zeros((d4, alpha91_closeadj.shape[1]))
    for ii in range(d4):
        jj = ii + d3 
        alpha91_rank1_decay[ii] = self._decay_linear(alpha91_rank1_decay_decay[ii:jj])

    alpha91_rank1 = self._ts_rank(alpha91_rank1_decay)


    alpha91_rank2_adv30 = np.zeros((d5+d6-1, alpha91_closeadj.shape[1]))
    for ii in range(d5+d6-1):
        jj = ii + d0
        alpha91_rank2_adv30[ii] = np.nanmean(alpha91_volume2, axis=0, keepdims=True)

    alpha91_rank2_corr = np.zeros((d6,alpha91_closeadj.shape[1]))
    for ii in range(d6):
        jj = ii + d5 
        alpha91_rank2_corr[ii] = self._correlation(alpha91_vwapadj[ii:jj], alpha91_rank2_adv30[ii:jj])

    alpha91_rank2 = self._rank(self._decay_linear(alpha91_rank2_corr))	

    alpha91 = (alpha91_rank1 - alpha91_rank2) * -1
    # 测试买卖反向 
    # alpha91 = alpha91 * -1 
    alpha = alpha91[0,:] * self.Universe_one.iloc[i,:]


    return alpha


# ################################################################################
# Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
# Alpha92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 15), 19), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7))
# 时序排名1 & 时序排名2 均大
# 时序排名1（19天）：布朗值的15日加权平均：close比open大很多 --> 加权平均小 --> 排名大 --> 买入
# 时序排名2（7天）：相关系数的8日加权平均：最低价 与 10日平均交易俩相关性低 --> 加权平均数小 --> 排名大 --> 买入
# 近期大部分日期：close比open大很多     &     最低价 与 10日平均交易俩相关性低


# 回测结果
## cum_return_rate           1.41336
## final_return_rate         0.20031
## beta                     0.894682
## alpha                    0.105342
## sharpe ratio             0.478754
## information ratio        0.368869
## turnover rate            0.111277
## max drawdown             0.518945
##   drawdown start date  2017-12-05
##   drawdown end date          None
## fitness                  0.642333

## 15年收益上涨明显，17年收益下降
# ################################################################################
def modified_alpha92():

    d0 = 30 
    d1 = 15
    d2 = 19 
    d3 = 8
    d4 = 7 
    d5 = 7 

    # data used to compute alpha92_rank1 (left part)
    alpha92_highadj1 = self.Highprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]
    alpha92_lowadj1 = self.Lowprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] 
    alpha92_closeadj1 = self.Closeprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]
    alpha92_openadj1 = self.Openprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] 

    # data used to compute alpha92_rank2 (right part)
    alpha92_low = self.Lowprice[di-DELAY-(d3+d4-1+d5-1)+1:di-DELAY+1,:]
    alpha92_volume = self.Volume[di-DELAY-(d0+d3-1+d4-1+d5-1)+1:di-DELAY+1,:]

    # 计算alpha92_rank1
    alpha92_rank1_decay = np.zeros((d2, alpha92_highadj1.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha92_rank1_decay[ii] = self._decay_linear(((alpha92_highadj1 + alpha92_closeadj1 - alpha92_lowadj1/2) < (alpha92_lowadj1 + alpha92_openadj1))[ii:jj])

    alpha92_rank1 = self._ts_rank(alpha92_rank1_decay)


    # 逐层计算 alpha92_rank2 
    alpha92_rank2_adv30 = np.zeros((d3+d4-1+d5-1, alpha92_highadj1.shape[1]))
    for ii in range(d3+d4-1+d5-1):
        jj = ii + d0 
        alpha92_rank2_adv30[ii] = np.nanmean(alpha92_volume[ii:jj,:], axis=0, keepdims=True)

    alpha92_rank2_corr = np.zeros((d4+d5-1, alpha92_highadj1.shape[1]))
    for ii in range(d4+d5-1):
        jj = ii + d3
        alpha92_rank2_corr[ii] = self._correlation(self._rank(alpha92_low[ii:jj]), self._rank(alpha92_rank2_adv30[ii:jj]))

    alpha92_rank2_decay = np.zeros((d5, alpha92_highadj1.shape[1]))
    for ii in range(d5):
        jj = ii + 1 
        alpha92_rank2_decay[ii] = self._decay_linear(alpha92_rank2_corr[ii:jj])

    alpha92_rank2 = self._ts_rank(alpha92_rank2_decay)


    # 计算alpha92
    alpha92 = np.where(alpha92_rank1 < alpha92_rank2, alpha92_rank1, alpha92_rank2)
    alpha92 = alpha92.reshape((1,alpha92.shape[0]))
    alpha = alpha92[0,:] * self.Universe_one.iloc[i,:]


    return alpha 












# ################################################################################
# Alpha#93: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
# Alpha93: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17), 20), 8) / rank(decay_linear(delta(((close * w) + (vwap * (1 - w))), 3), 16)))


# 回测结果
# 代码valify 通过  by xx
# ################################################################################
def modified_alpha93():

    w = 0.524434
    d0 =81 
    d1 = 17 
    d2 = 20 
    d3 = 8 

    d4 = 3 
    d5 = 16

    Industry = self.Industry[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:,:]  

    alpha93_volume = self.Volume[di-DELAY-(d0+d1-1+d2-1+d3-1)+1:di-DELAY+1,:]
    alpha93_vwapadj = self.vwap[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:]
    alpha93_industry = self.Industry[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1,:]
    alpha93_Ind_vwapadj = self._industry_neutral(alpha93_vwapadj, alpha93_industry)

    alpha93_closeadj = self.Closeprice[di-DELAY-(d4+d5-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d4+d5-1)+1:di-DELAY+1,:]

    # rank1: Ts_Rank(decay...(correlation...))

    alpha93_rank1_adv81 = np.zeros((d1+d2-1+d3-1, alpha93_volume.shape[1]))
    for ii in range(d1+d2-1+d3-1):
        jj = ii + d0 
        alpha93_rank1_adv81[ii] = np.nanmean(alpha93_volume, axis=0, keepdims=True)

    alpha_rank1_corr = np.zeros((d2+d3-1, alpha93_volume.shape[1]))
    for ii in range(d2+d3-1):
        jj = ii + d1
        alpha_rank1_corr[ii] = self._correlation(alpha93_Ind_vwapadj[ii:jj], alpha93_rank1_adv81[ii:jj])

    alpha_rank1_decay = np.zeros((d3, alpha93_volume.shape[1]))
    for ii in range(d3):
        jj = ii + d2
        alpha_rank1_decay[ii] = self._decay_linear(alpha_rank1_corr[ii:jj])	

    alpha93_rank1 = self._ts_rank(alpha_rank1_decay)

    # rank2: rank(decay...)

    alpha93_rank2_delta = np.zeros((d5, alpha93_vwapadj.shape[1]))
    for ii in range(d5):
        jj = ii + d4
        alpha93_rank2_delta[ii] = self._delta((alpha93_closeadj * w + alpha93_vwapadj[-(d4+d5-1):,:]*(1-w))[ii:jj])

    alpha93_rank2 = self._rank(self._decay_linear(alpha93_rank2_delta))


    # alpha93 = alpha93_rank1 / alpha93_rank2 
    # 测试反向
    alpha93 = alpha93_rank2 / alpha93_rank1
    alpha = alpha93[0,:] * self.Universe_one.iloc[i,:]


    return alpha













# ################################################################################
# Alpha#94: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
#Alpha94: ((rank((vwap - ts_min(vwap, 12)))^Ts_Rank(correlation(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3)) * -1)
# 卖出策略：横截面上X排序 ^ 时间序列上Y排序
# X：今日平均价与12天前最低平均价的价差：价差越小，rank排名越大，卖出
# Y：时序排序3天：Y1与Y2的相关系数：相关性越低，时序排名越高，卖出
# Y1——今天vwap在过去20天内的时序排名；Y2——平均交易量60在过去4天的时序排名
# 1. 今日均价越低，卖出  2. 均价时序排名 与 平均交易量时序排名相关性弱，则卖出

# 回测结果
## cum_return_rate         -0.778256
## final_return_rate       -0.268126
## beta                    -0.860073
## alpha                   -0.246402
## sharpe ratio            -0.921178
## information ratio       -0.709057
## turnover rate           0.0696821
## max drawdown             0.850923
##   drawdown start date  2016-11-22
##   drawdown end date          None
## fitness                  -1.80697

## 建议：买卖反向
# ################################################################################
def modified_alpha94():


    d0 = 60
    d1 = 20
    d2 = 4 
    d3 = 18 
    d4 = 3 

    alpha94_vwapadj = self.vwap[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:]
    alpha94_volume = self.Volume[di-DELAY-(d0+d1-1+d2-1+d3-1+d4-1)+1:di-DELAY+1,:]



    # rank1 计算
    alpha94_rank1 = self._rank(alpha94_vwapadj[d1-1:d1,:] - self._ts_min(alpha94_vwapadj[-12:,:])) # 只需要最后12天的vwap 即可

    # rank2 计算 逐层向外计算
    alpha94_rank2_adv60 = np.zeros((d2+d3-1+d4-1, alpha94_vwapadj.shape[1]))
    for ii in range(d2+d3-1+d4-1):
        jj = ii + d0
        alpha94_rank2_adv60[ii] = np.nanmean(alpha94_volume[ii:jj], axis=0, keepdims=True)

    alpha94_rank2_tsrank1 = np.zeros((d3+d4-1,alpha94_vwapadj.shape[1]))
    alpha94_rank2_tsrank2 = np.zeros((d3+d4-1,alpha94_vwapadj.shape[1]))

    for ii in range(d3+d4-1):
        jj1 = ii + d1 
        jj2 = ii + d2
        alpha94_rank2_tsrank1[ii] = self._ts_rank(alpha94_vwapadj[ii:jj1,:])
        alpha94_rank2_tsrank2[ii] = self._ts_rank(alpha94_rank2_adv60[ii:jj2,:])

    alpha94_rank2_corr = np.zeros((d4,alpha94_vwapadj.shape[1]))
    for ii in range(d4):
        jj = ii + d3
        alpha94_rank2_corr[ii] = self._correlation(alpha94_rank2_tsrank1[ii:jj,:], alpha94_rank2_tsrank2[ii:jj,:])

    alpha94_rank2  = self._ts_rank(alpha94_rank2_corr)

    alpha94 = (alpha94_rank1 ** alpha94_rank2) * -1
    alpha = alpha94[0,:] * self.Universe_one.iloc[i,:]


    return alpha 


# ################################################################################
# Alpha#95: (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
#Alpha95: (rank((open - ts_min(open, 12))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19), sum(adv40, 19), 13))^5), 12))
# 买入策略：若横截面排序X 小于 时序排序Y 则买入。否则alpha = 0
# X：横截面排序rank1：今日开盘价与过去12天最低开盘价的家价差
#今日开盘价越高，价差越高，X排序越小，买入

# Y：时序排序（12天）：横截面排序rank2的5次方
# rank2 横截面排序：Y1 与 Y2 的相关系数（13天）
# Y1：过去19天 (high + low) / 2) 的加和
# Y2：过去19天 40日平均交易量 的加和
# (high + low) / 2) 与 40日平均交易量 相关性低，相关系数小 --> rank2小 --> 时序排序大 --> 买入

# 今日开盘价越高 & (high + low) / 2) 与 40日平均交易量 相关性低


# 回测结果
##p_mean                  0.0518287
##t_mean                    2.00484
##t_std                    0.294928
##t_sig_prt              0.00411862
##t_2                       235.542
##p_2                             0
##R_squared_adj_mean      0.0102425
##R_squared_adj_std        0.496941
##IC_mean                0.00360335
##IC_abs_mean             0.0403898
##IC_std                   0.497726
##IC_positive_prt          0.997529
##IR                        12.3036

##15年收益明显上涨，17年收益下降
# ################################################################################

def modified_alpha95():

    d0 = 40
    d1 = 12 
    d2 = 19
    d3 = 13 

    alpha95_openadj = self.Openprice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:] 
    alpha95_highadj = self.Highprice[di-DELAY-(d2+d1-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2+d1-1)+1:di-DELAY+1,:]
    alpha95_lowadj = self.Lowprice[di-DELAY-(d2+d1-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2+d1-1)+1:di-DELAY+1,:]
    alpha95_volume = self.Volume[di-DELAY-(d0+d2-1+d3-1+d1-1)-1:di-DELAY+1, :] 

    # 计算rank1 
    alpha95_rank1 = self._rank(alpha95_openadj[d1-1:d1,:] - self._ts_min(alpha95_openadj))

    # 计算rank2, 逐层计算，由内向外

    alpha95_rank2_adv40 = np.zeros((d2+d3-1+d1-1,alpha95_openadj.shape[1]))
    for ii in range(d2+d3-1+d1-1):
        jj = ii + d0 
        alpha95_rank2_adv40[ii] = np.nanmean(alpha95_volume[ii:jj,:], axis=0, keepdims=True)

    alpha95_sumadv40 = np.zeros((d3+d1-1, alpha95_openadj.shape[1]))
    for ii in range(d3+d1-1):
        jj = ii + d2 
        alpha95_sumadv40[ii] = np.nansum(alpha95_rank2_adv40, axis=0, keepdims=True)


    alpha95_rank2_sum1 = np.zeros((d2+d1-1, alpha95_openadj.shape[1]))
    alpha95_rank2_sum2 = np.zeros((d2+d1-1, alpha95_openadj.shape[1]))
    for ii in range(d2+d1-1):
        jj = ii + d2 
        alpha95_rank2_sum1[ii] = np.nansum((alpha95_lowadj[ii:jj,:]+alpha95_highadj[ii:jj,:])/2, axis=0, keepdims=True)
        alpha95_rank2_sum2[ii] = np.nansum(alpha95_sumadv40[ii:jj,:], axis=0, keepdims=True)

    alpha95_rank2_corr = np.zeros((d1, alpha95_openadj.shape[1]))
    for ii in range(d1):
        jj = ii + d2 
        alpha95_rank2_corr[ii] = self._correlation(alpha95_rank2_sum1[ii:jj,:], alpha95_rank2_sum2[ii:jj,:])

    alpha95_rank2 = self._ts_rank(self._rank(alpha95_rank2_corr))

    alpha95 = alpha95_rank1 < alpha95_rank2 

    alpha = alpha95[0,:] * self.Universe_one.iloc[i,:]

    return alpha 








# ################################################################################
# Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
# Alpha96: ( max(              
#                 Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8), 
#                 Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7), Ts_Rank(adv60, 4), 4), 13), 14), 13)
#               ) 
#             * -1
#          )

# 回测结果
# 代码 valify 通过 by xx
# ################################################################################

def modified_alpha96():

    d1 = 4 
    d2 = 4 
    d3 = 8

    d0 = 60 
    d4 = 7 
    d5 = 13 
    d6 = 14 
    d7 = 13 


    alpha96_vwapadj = self.vwap[di-DELAY-(d1+d2-1+d3-1):di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1+d3-1):di-DELAY+1,:]
    alpha96_volume = self.Volume[di-DELAY-(d0+d1-1+d2-1+d5-1+d6-1+d7-1):di-DELAY+1,:]

    alpha96_closeadj = self.Closeprice[di-DELAY-(d4+d2-1+d5-1+d6-1+d7-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d4+d2-1+d5-1+d6-1+d7-1)+1:di-DELAY+1,:]
    #alpha96_rank1:  Ts_Rank(decay_linear(correlation...))

    alpha96_volume_length = alpha96_volume.shape[0]
    alpha96_volume_rank1 = alpha96_volume[alpha96_volume_length-(d1+d2-1+d3-1)-1:alpha96_volume_length,:]

    alpha96_rank1_corr = np.zeros((d2+d3-1, alpha96_vwapadj.shape[1]))
    for ii in range(d2+d3-1):
        jj = ii + d1 
        alpha96_rank1_corr[ii] = self._correlation(self._rank(alpha96_vwapadj[ii:jj]), self._rank(alpha96_volume_rank1[ii:jj]))


    alpha96_rank1_decay = np.zeros((d3, alpha96_vwapadj.shape[1]))
    for ii in range(d3):
        jj = ii + d2 
        alpha96_rank1_decay[ii] = self._ts_argmax(alpha96_rank1_corr[ii:jj]) 

    alpha96_rank1 = self._ts_rank(alpha96_rank1_decay) # To-do


    #alpha96_rank2:  Ts_rank(decay_linear(Ts_ArgMax...)) 
    alpha96_rank2_adv60 = np.zeros((d1+d2-1+d5-1+d6-1+d7-1, alpha96_vwapadj.shape[1]))
    for ii in range(d1+d2-1+d5-1+d6-1+d7-1):
        jj = ii + d0 
        alpha96_rank2_adv60[ii] = np.nanmean(alpha96_volume[ii:jj], axis=0, keepdims=True)

    alpha96_rank2_tsrank_adv60 = np.zeros((d2+d5-1+d6-1+d7-1, alpha96_vwapadj.shape[1]))
    alpha96_rank2_tsrank_close =  np.zeros((d2+d5-1+d6-1+d7-1, alpha96_vwapadj.shape[1]))	

    for ii in range(d2-1+d5-1+d6-1+d7-1):
        jj = ii + d1 
        jj2 = ii + d4 
        alpha96_rank2_tsrank_adv60[ii] = self._ts_rank(alpha96_rank2_adv60[ii:jj])
        alpha96_rank2_tsrank_close[ii] = self._ts_rank(alpha96_closeadj[ii:jj2])


    # alpha96_volume_length = alpha96_volume.shape[0]
    # alpha96_volume_rank1 = alpha96_volume[alpha96_volume_length-(d1+d2-1+d3-1)-1:alpha96_volume_length,:]
    # alpha96_closeadj_length = alpha96_closeadj.shape[0]
    # alpha96_rank2_tsrank_close = alpha96_closeadj[alpha96_closeadj_length-(d2+d5-1+d6-1+d7-1)-1:alpha96_closeadj_length,:]

    alpha96_rank2_corr = np.zeros((d5+d6-1+d7-1,alpha96_vwapadj.shape[1]))
    for ii in range(d5+d6-1+d7-1):
        jj = ii + d2 
        alpha96_rank2_corr[ii] = self._correlation(alpha96_rank2_tsrank_close[ii:jj], alpha96_rank2_tsrank_adv60[ii:jj])  

    alpha96_rank2_tsargmax = np.zeros((d6+d7-1, alpha96_vwapadj.shape[1]))
    for ii in range(d6+d7-1):
        jj = ii + d5
        alpha96_rank2_tsargmax[ii] = self._ts_argmax(alpha96_rank2_corr[ii:jj])

    alpha96_rank2_decay = np.zeros((d7, alpha96_vwapadj.shape[1]))
    for ii in range(d7):
        jj = ii + d6 
        alpha96_rank2_decay[ii] = self._decay_linear(alpha96_rank2_tsargmax[ii:jj])

    alpha96_rank2 = self._ts_rank(alpha96_rank2_decay)	

    alpha96 = np.nanmax(np.row_stack((alpha96_rank1, alpha96_rank2)), axis = 0) * -1 
    # 测试买卖反向 
    # alpah96 = alpha96 * -1 
    alpha = alpha96 * self.Universe_one.iloc[i,:]


    return alpha






# ################################################################################
# Alpha#97: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
#Alpha97 = ((rank(decay_linear(delta(IndNeutralize(((low * w) + (vwap * (1 - w))), IndClass.industry), 3), 20)) - 
#            Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7), Ts_Rank(adv60, 17), 5), 19), 16), 7)) * -1)


# 回测结果
# valify 通过验证(by xx)
# ################################################################################
def modified_alpha97():

    d1 = 3 
    d2 = 20 

    d0 = 60
    d3 = 7 
    d4 = 17
    d5 = 5
    d6 = 19
    d7 =16
    d8 = 7 

    w = 0.721001

    # 用于 计算 rank(decay(...)) 的数据
    Industry = self.Industry[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:,:]
    alpha97_lwoadj = self.Lowprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1, :] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1, :]
    alpha97_vwapadj = self.vwap[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1, :]

    # 用于计算ts_rank(decay...) 的数据
    alpha97_lowadj2 = self.Lowprice[di-DELAY-(d3+d5-1+d6-1+d7-1+d8-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d3+d5-1+d6-1+d7-1+d8-1)+1:di-DELAY+1,:]
    alpha97_volume = self.Volume[di-DELAY-(d0+d4-1+d5-1+d6-1+d7-1+d8-1)+1:di-DELAY+1,:]


    # 计算alpha97_rank1 
    indneutural_price = self._industry_neutral((alpha97_lwoadj * w + alpha97_vwapadj * (1-w)) , Industry) # 计算行业中心化的价格

    alpha97_rank1_delta = np.zeros((d2,alpha97_lwoadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha97_rank1_delta[ii] = self._delta(indneutural_price[ii:jj])

    alpha97_rank1 = self._rank(self._decay_linear(alpha97_rank1_delta))


    # 计算rank2

    alpha97_rank2_corr_tsrank1 = np.zeros((d5+d6-1+d7-1+d8-1,alpha97_lwoadj.shape[1]))
    for ii in range(d5+d6-1+d7-1+d8-1):
        jj = ii + d3 
        alpha97_rank2_corr_tsrank1[ii] = self._ts_rank(alpha97_lowadj2[ii:jj])


    alpha97_rank2_corr_adv60 = np.zeros((d4+d5-1+d6-1+d7-1+d8-1,alpha97_lwoadj.shape[1]))
    for ii in range(d4+d5-1+d6-1+d7-1+d8-1):
        jj = ii + d0
        alpha97_rank2_corr_adv60[ii] = np.nanmean(alpha97_volume[ii:jj], axis=0, keepdims=True)

    alpha97_rank2_corr_tsrank2 = np.zeros((d5+d6-1+d7-1+d8-1,alpha97_lwoadj.shape[1]))
    for ii in range(d5+d6-1+d7-1+d8-1):
        jj = ii + d4
        alpha97_rank2_corr_tsrank2[ii] = self._ts_rank(alpha97_alpha07_rank2_corr_adv60[ii:jj])

    alpha97_rank2_corr = np.zeros((d6+d7-1+d8-1,alpha97_lwoadj.shape[1]))
    for ii in range(d6+d7-1+d8-1):
        jj = ii + d5 
        alpha97_rank2_corr[ii] = self._correlation(alpha97_rank2_corr_tsrank1[ii:jj], alpha97_rank2_corr_tsrank2[ii:jj])

    alpha97_rank2_tsrank = np.zeros((d7+d8-1,alpha97_lwoadj.shape[1])) # 这个是第二部分ts_rank中的第二层ts_rank(corr...)
    for ii in range(d7+d8-1):
        jj = ii + d6
        alpha97_rank2_tsrank[ii] = self._ts_rank(alpha97_rank2_corr[ii:jj])

    alpha97_rank2_decay = np.zeros((d8,alpha97_lwoadj.shape[1]))
    for ii in range(d8):
        jj = ii + d7 
        alpha97_rank2_decay[ii] = self._decay_linear(alpha97_rank2_tsrank[ii:jj])

    alpha97_rank2 = self._ts_rank(alpha97_rank2_decay)

    alpha97 = (alpha97_rank1 - alpha97_rank2 ) * -1


    alpha = alpha97[0,] * self.Universe_one.iloc[i,:]


    return alpha


# rank1 - rank2
# rank1: 过去7天相关系数1加权平均
# 相关系数1（7天）：均价 & 过去26天的5日平均交易量加和
# 近期：均价 与 过去一个月5日平均交易量相关性越低 --> 相关系数小 --> rank1 大 --> 买入

# rank2：过去8天时序排序2的加权平均
# 时序排序2（7天）：过去9天中相关系数2最小值发生的日期
# 相关系数2（21天）：今日开盘价 & 15日平均交易量
# 近期：开盘价 与 交易量相关性越低 --> 相关系数小 --> 最小相关系数发生日期离今天越近 --> argmin越大 --> 时序排序越小 --> 加权平均越小 --> rank2越大 --> 不买入

# 买入：近期：均价 与 过去一个月5日平均交易量相关性越低     &     开盘价 与 交易量相关性越高
# 回测结果
##p_mean                   0.379811
##t_mean                    1.21546
##t_std                     1.57829
##t_sig_prt                0.170511
##t_2                       1.02664
##p_2                      0.304691
##R_squared_adj_mean      0.0161776
##R_squared_adj_std      0.00568643
##IC_mean                 0.0329302
##IC_abs_mean             0.0812201
##IC_std                  0.0635485
##IC_positive_prt          0.512356
##IR                      0.0700125

##15年收益无明显上涨，其余年份均与指数收益相近
# ################################################################################
def modified_alpha98():
    d0 = 5 
    d1 = 26
    d2 = 5 
    d3 = 7

    d4 = 15 
    d5 = 21 
    d6 = 9 
    d7 = 7 
    d8 = 8 

    # 计算rank1 的数据
    alpha98_vwapadj = self.vwap[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2+d3-1)+1:di-DELAY+1,:]
    alpha98_volume = self.Volume[di-DELAY-(d0+d1-1+d2-1+d3-1)+1:di-DELAY+1,:]
    # 计算rank2 的数据
    alpha98_open = self.Openprice[di-DELAY-(d5+d6-1+d7-1+d8-1)+1:di-DELAY+1,:]
    alpha98_volume2 = self.Volume[di-DELAY-(d4+d5-1+d6-1+d7-1+d8-1)+1:di-DELAY+1,:]

    # 计算rank1 逐层计算，从内向外
    alpha98_rank1_adv5 = np.zeros((d1+d2-1+d3-1, alpha98_vwapadj.shape[1]))
    for ii in range(d1+d2-1+d3-1):
        jj = ii + d0 
        alpha98_rank1_adv5[ii] = np.nanmean(alpha98_volume[ii:jj,:], axis=0, keepdims=True)

    alpha98_rank1_sumadv = np.zeros((d2+d3-1,alpha98_vwapadj.shape[1]))
    for ii in range(d2+d3-1):
        jj = ii + d1 
        alpha98_rank1_sumadv[ii] = np.nansum(alpha98_rank1_adv5[ii:jj], axis=0, keepdims=True)


    alpha98_rank1_corr = np.zeros((d3,alpha98_vwapadj.shape[1]))
    for ii in range(d3):
        jj = ii+d2
        alpha98_rank1_corr[ii] = self._correlation(alpha98_vwapadj[ii:jj,:], alpha98_rank1_sumadv[ii:jj,:])

    alpha98_rank1 = self._rank(self._decay_linear(alpha98_rank1_corr)) 


    # 计算rank2 逐层计算，从内向外
    alpha98_rank2_adv15 = np.zeros((d5+d6-1+d7-1+d8-1,alpha98_vwapadj.shape[1]))
    for ii in range(d5+d6-1+d7-1+d8-1):
        jj = ii + d4 
        alpha98_rank2_adv15[ii] = np.nanmean(alpha98_volume2[ii:jj,:], axis=0, keepdims=True)

    alpha98_rank2_corr = np.zeros((d6+d7-1+d8-1,alpha98_vwapadj.shape[1])) 
    for ii in range(d6+d7-1+d8-1):
        jj = ii + d5
        alpha98_rank2_corr[ii] = self._correlation(self._rank(alpha98_open[ii:jj,:]), self._rank(alpha98_rank2_adv15[ii:jj,:]))

    alpha98_rank2_ts_argmin = np.zeros((d7+d8-1,alpha98_vwapadj.shape[1]))
    for ii in range(d7+d8-1):
        jj = ii + d6
        alpha98_rank2_ts_argmin[ii] = self._ts_argmin(alpha98_rank2_corr[ii:jj,:])

    alpha98_rank2_tsrank = np.zeros((d8,alpha98_vwapadj.shape[1]))
    for ii in range(d8):
        jj = ii + d7
        alpha98_rank2_tsrank[ii] = self._ts_rank(alpha98_rank2_ts_argmin[ii:jj,:])

    alpha98_rank2 = self._rank(self._decay_linear(alpha98_rank2_tsrank))  

    # 计算alpha指标
    alpha98 = alpha98_rank1 - alpha98_rank2 
    alpha = alpha98[0,:] * self.Universe_one.iloc[i,:]


    return alpha 


# ################################################################################
# Alpha#99: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
# Alpha99: ((rank(correlation(sum(((high + low) / 2), 20), sum(adv60, 20), 9)) < rank(correlation(low, volume, 6))) * -1)
# 卖出策略：
# 等权卖出符合条件的股票：相关系数1rank < 相关系数2rank：Y1与Y2相关性  高于  X1与X2相关性
# Y1：过去20天(high+low)/2的加和，Y2：过去20天adv60（60天交易量平均数）的加和
# X1：最低价，X2：交易量
#意义：卖出最低价与交易量高度相关的股票
#（相关程度的衡量方法：1.与股票过去20天均值比较 2.相关系数横截面排序）

# 回测结果
##cum_return_rate         -0.798353
##final_return_rate       -0.282395
##beta                     -1.00985
##alpha                   -0.250711
##sharpe ratio            -0.807895
##information ratio       -0.662446
##turnover rate            0.152288
##max drawdown             0.854748
    ##drawdown start date  2016-11-22
    ##drawdown end date          None
##fitness                  -1.10015

#建议：买卖反向

# ################################################################################
def modified_alpha99():
    d0 = 60
    d1 = 20
    d2 = 9

    d3 = 6 

    alpha99_highadj = self.Highprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]
    alpha99_lowadj = self.Lowprice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:]
    alpha99_volume = self.Volume[di-DELAY-(d0+d1-1+d2-1)+1:di-DELAY+1,:]

    # 计算rank1 由内向外 逐层计算
    alpha99_rank1_sumprice = np.zeros((d2, alpha99_highadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha99_rank1_sumprice[ii] = np.nansum((alpha99_highadj[ii:jj,:]+alpha99_lowadj[ii:jj,:])/2, axis=0, keepdims=True)

    alpha99_rank1_adv60 = np.zeros((d1+d2-1, alpha99_highadj.shape[1]))
    for ii in range(d1+d2-1):
        jj = ii + d0 
        alpha99_rank1_adv60[ii] = np.nanmean(alpha99_volume[ii:jj,:], axis=0, keepdims=True)

    alpha99_rank1_sumadv = np.zeros((d2, alpha99_highadj.shape[1]))
    for ii in range(d2):
        jj = ii + d1 
        alpha99_rank1_sumadv[ii] = np.nansum(alpha99_rank1_adv60[ii:jj,:], axis=0, keepdims=True)

    alpha99_rank1 = self._rank(self._correlation(alpha99_rank1_sumprice, alpha99_rank1_sumadv)) 

    # 计算 rank2 
    alpha99_rank2 = self._rank(self._correlation(alpha99_lowadj[-d3:, :], alpha99_volume[-d3:, :]))

    alpha99 = (alpha99_rank1 < alpha99_rank2) * -1
    # 买卖反向
    alpha99 = alpha99 * -1
    alpha = alpha99[0,:] * self.Universe_one.iloc[i,:]

    return alpha 


# ################################################################################
# Alpha#100: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
# Alpha100: (0 - 
#               (1 * 
#                   (    (
#                            (   1.5 
#                                * 
#                                scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) 
#                            - 
#                            scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))
#                        ) 
#                        * 
#                        (volume / adv20)
#                   )
#               )
#            )


# 回测结果
# valify 运行通过 (by xx)
# ################################################################################
def modified_alpha100():

    d0=20
    d1 = 5 
    d2 = 30 

    alpha100_closeadj  = self.Closeprice[di-DELAY-(d2)+1:di-DELAY+1] * self.adjfactor[di-DELAY-d2+1:di-DELAY+1]
    alpha100_close = self.Closeprice[di-DELAY:di-DELAY+1]
    alpha100_low = self.Lowprice[di-DELAY:di-DELAY+1]
    alpha100_high = self.Highprice[di-DELAY:di-DELAY+1]
    alpha100_volume = self.Volume[di-DELAY-(d0+d1-1)+1:di-DELAY+1]


    #计算scale1
    Industry = self.Industry[di-DELAY:di-DELAY+1,:,:]
    alpha100_scale1_measure = ((alpha100_close * 2 -alpha100_high - alpha100_low) / (alpha100_high - alpha100_low) * alpha100_volume[d0+d1-2:d0+d1-1])
    alpha100_scale1 = self._scale(self._industry_neutral(self._rank(alpha100_scale1_measure), Industry)) 


    # 计算scale2 
    alpha100_scale2_corr_adv20 = np.zeros((d1, alpha100_close.shape[1]))
    for ii in range(d1):
        jj = ii + d0 
        alpha100_scale2_corr_adv20[ii] = np.nanmean(alpha100_volume, axis=0, keepdims=True)

    alpha100_scale2_corr_rankadv = self._rank(alpha100_scale2_corr_adv20)
    alpha100_scale2_corr = self._correlation(alpha100_closeadj[-5:,:], alpha100_scale2_corr_rankadv)
    alpha100_scale2_rank = self._rank(self._ts_argmin(alpha100_closeadj))

    alpha100_scale2 = self._scale(self._industry_neutral((alpha100_scale2_corr-alpha100_scale2_rank), Industry))

    alpha100 = 0- (1.5 * alpha100_scale1 - alpha100_scale2) * alpha100_volume[d0+d1-2:d0+d1-1]/np.nanmean(alpha100_volume[-d0:,:], axis=0, keepdims=True)

    alpha = alpha100[0,:] * self.Universe_one.iloc[i,:] 




    return alpha 





# ################################################################################
# Alpha#101: ((close - open) / ((high - low) + .001))


# 回测结果
# 近期一致走弱 
##p_mean                    0.242778
##t_mean                     2.15771
##t_std                      2.91548
##t_sig_prt                 0.415157
##t_2                       -6.78491
##p_2                     1.4546e-11
##R_squared_adj_mean       0.0287641
##R_squared_adj_std       -0.0347906
##IC_mean                  0.0458035
##IC_abs_mean               0.149575
##IC_std                     0.11765
##IC_positive_prt           0.381384
##IR                       -0.232595
# ################################################################################
def modified_alpha101():
    alpha101_open = self.Openprice[di-DELAY:di-DELAY+1,:]
    alpha101_close = self.Closeprice[di-DELAY:di-DELAY+1,:]
    alpha101_high = self.Highprice[di-DELAY:di-DELAY+1,:]
    alpha101_low = self.Lowprice[di-DELAY:di-DELAY+1,:]

    alpha101 = (alpha101_close-alpha101_open)/(alpha101_high-alpha101_low + 0.001)
    alpha = alpha101[0,:] * self.Universe_one.iloc[i,:]

    return alpha

