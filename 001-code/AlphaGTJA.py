

import pandas as pd 
import numpy as np 



def GJ_alpha175(self):
	# #############################################################
	# GJ_alpha# :MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
	# 含义 过去6天内 的价差 -- 衡量可能的上涨强度 
	#
	# 策略方向：
	# 主要买入：
	# 主要卖出：
	#---------------------------------------------
	# 评价：如果把这个看成是短期向上增长的潜力 
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	d1 = 6 + 1
	
	GJ175_highadj = self.HighPrice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
	GJ175_lowadj = self.LowPrice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
	GJ175_closeadj = self.ClosePrice[di-DELAY-(d1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1)+1:di-DELAY+1,:]
	
	GJ175_price_range = (GJ175_highadj-GJ175_lowadj)[1:,:]
	GJ175_abs1 = np.abs(GJ175_closeadj[:-1,:] - GJ175_highadj[1:,:])
	GJ175_abs2 = np.abs(GJ175_closeadj[:-1,:] - GJ175_lowadj[1:,:])
	
	max_value = (np.where(GJ175_abs1>GJ175_abs2, GJ175_abs1, GJ175_abs2))
	
	GJ175_max= self._ts_max(np.where(GJ175_price_range>max_value, GJ175_price_range, max_value))
	
	GJ175 = np.nanmean(GJ175_max, axis=0, keepdims=True) 
	# 调整，将价差用过去一段时间的收盘价做标准化
	GJ175 = np.nanmean(GJ175_max, axis=0, keepdims=True) / np.mean(GJ175_closeadj, axis=0, keepdims=True) 
	
	alpha = GJ175[0,:] * self.Universe_one.iloc[i,:]
	return alpha 



def GJ_alpha177_totest(self):
	# #############################################################
	# GJ_alpha#177:((20-HIGHDAY(HIGH,20))/20)*100
	# 含义
	#
	# 策略方向：
	# 主要买入：
	# 主要卖出：
	#---------------------------------------------
	# 评价
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	DELAY = self.DELAY

	d = 20 
	GJ171_highadj = self.HighPrice[di-DELAY-d+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d+1:di-DELAY+1,:]

	GJ171 = (20 - self._ts_argmax(GJ171_highadj))/20 * 100
	GJ171= 1/GJ171
	alpha = GJ171[0,:] * self.Universe_one.iloc[i,:]
	return alpha 



def GJ_alpha178_totest(self):
	# #############################################################
	# GJ178:(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
	# 含义 return * volume 
	#
	# 策略方向：买入
	# 主要买入：
	# 主要卖出：
	#---------------------------------------------
	# 评价
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	
	d = 2 

	GJ178_closeadj = self.ClosePrice[di-DELAY-d+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d+1:di-DELAY+1,:]
	GJ178_volume = self.Volume[di-DELAY-d+1:di-DELAY+1,:] 

	GJ178 = np.diff(GJ178_closeadj, axis=0) / GJ178_closeadj[0:(d-1),:] * GJ178_volume[1:2,:]

	alpha = GJ178[0,:] * self.Universe_one.iloc[i,:]
	return alpha 



def GJ_alpha181_totest(self):
	# #############################################################
	# GJ181 SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)
	# 含义 20天累加[(return - 2天 return均值) - (index - 20天index均值)^2] / 20天累加[(index - 20天平均index)^3]
	# 
	# 策略方向：
	# 主要买入：
	# 主要卖出：
	#---------------------------------------------
	# 评价: 各个相加减的指标不在一个数量级上？
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	
	DELAY = self.DELAY

	d1 = 20 + 1
	d2 = 20
	GJ181_closeadj = self.ClosePrice[di-DELAY-d1+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-d1+1:di-DELAY+1,:]
	GJ181_ret = np.diff(GJ181_closeadj, axis=0)/GJ181_closeadj[:-1,:]
	GJ181_index_close = self.HS300[di-DELAY-d2+1:di-DELAY+1,:]

	GJ181_demean_ret = GJ181_ret - np.nanmean(GJ181_ret, axis=0, keepdims=True) 
	GJ181_demean_index = GJ181_index_close - np.nanmean(GJ181_index_close, axis=0, keepdims=True) 

	GJ181 = np.nansum(GJ181_demean_ret-np.power(GJ181_demean_index,2), axis=0, keepdims=True) / np.nansum(np.power(GJ181_demean_index,3), axis=0, keepdims=True)
	GJ181 = 1/GJ181
	alpha = GJ181[0,:] * self.Universe_one.iloc[i,:]
	return alpha 






def GJ_alpha183_totest(self):
	# #############################################################
	# GJ183 MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
	# 含义：过去24天内 close价格相对均值的变动幅度越大 
	# 
	# 策略方向：买入策略
	# 主要买入：买入波动幅度较大的股票 
	# 主要卖出：
	#---------------------------------------------
	# 评价
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	DELAY = self.DELAY


	d = 24 
	GJ183_closeadj = self.ClosePrice[di-DELAY-(d)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d)+1:di-DELAY+1,:]

	GJ183_data = np.nancumsum((GJ183_closeadj-np.nanmean(GJ183_closeadj, axis=0, keepdims=True)), axis=0) 
	GJ183_std = np.nanstd(GJ183_closeadj, axis=0, keepdims=True)

	GJ183 = ( np.nanmax(GJ183_data, axis=0, keepdims=True) - np.nanmin(GJ183_data, axis=0, keepdims=True) ) / GJ183_std 

	alpha = GJ183[0,:] * self.Universe_one.iloc[i,:]
	return alpha




def GJ_alpha185_totest(self):
	# #############################################################
	# GJ_alpha#185 : RANK((-1 * ((1 -(OPEN / CLOSE))^2)))
	# 含义: 日内价差收益率的平方 排序 
	#
	# 策略方向：
	# 主要买入：
	# 主要卖出：
	#---------------------------------------------
	# 评价：买卖日内收益率的波动？波动大/小的买入较多
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	GJ185_open = self.ClosePrice[di-DELAY:di-DELAY+1]
	GJ185_close = self.ClosePrice[di-DELAY:di-DELAY+1]
	
	GJ185 = self._rank(-1*(np.power((1-GJ185_open/GJ185_close),2)))
	
	alpha = GJ185[0,:] * self.Universe_one.iloc[i,:]
	return alpha 





def GJ_alpha186_totest(self):
	# #############################################################
	
	#(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
	
	# GJ#186 = (
	#              MEAN(
	#                   ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14)) 
	#                   /
	#                   (SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100
	#                 ,6)
	#                +
	#             DELAY(
	#                MEAN(
	#                  ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))
	#                      /
	#                     (SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100
	#               ,6)
	#             ,6)
	#         )/2
	# 含义
	#
	# 策略方向：
	# 主要买入：
	# 主要卖出：
	#---------------------------------------------
	# 评价
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	DELAY = self.DELAY

	d1 = 14 + 1 # 加1 是因为 LD, HD, TR 这些指标都需要1阶滞后的数值
	d2 = 6 
	d3 = 6 

	GJ186_lowadj = self.LowPrice[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1, :] * self.adjfactor[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1, :] 
	GJ186_highadj = self.HighPrice[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1, :] * self.adjfactor[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1, :] 
	GJ186_closeadj = self.ClosePrice[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1, :] * self.adjfactor[di-DELAY-(d1+d2-1+d3-1)+1:di-DELAY+1, :] 


	GJ186_LD = -1 * np.diff(GJ186_lowadj, axis=0)
	GJ186_HD = -1 * np.diff(GJ186_highadj, axis=0)

	GJ186_TR1 = (GJ186_highadj - GJ186_lowadj)[1:,:] 
	GJ186_TR2 = np.abs(GJ186_highadj[1:,:] - GJ186_closeadj[:-1,:])
	GJ186_TR3 = np.abs(GJ186_lowadj[1:,:] - GJ186_closeadj[:-1,:] )
	GJ186_TR12 = GJ186_TR1 * (GJ186_TR1>GJ186_TR2) + GJ186_TR2 * (GJ186_TR1<=GJ186_TR2)
	GJ186_TR = GJ186_TR12 * (GJ186_TR12>GJ186_TR3) + GJ186_TR3 * (GJ186_TR12<=GJ186_TR3)



	df_GJ186_base1 = pd.DataFrame( GJ186_LD * ((GJ186_LD>0) & (GJ186_LD > GJ186_HD)) )
	df_GJ186_base2 = pd.DataFrame( GJ186_HD * ((GJ186_HD>0) & (GJ186_HD > GJ186_LD)) )
	df_GJ186_base3 = pd.DataFrame( GJ186_TR )

	# 累积滚动计算 Mean(...) 中的内容
	df_GJ186_base = ((df_GJ186_base1.rolling(14,2).sum()*100 - df_GJ186_base2.rolling(14,2).sum()*100) / (df_GJ186_base1.rolling(14,2).sum()*100 + df_GJ186_base2.rolling(14,2).sum()*100)).rolling(6,2).mean()

	GJ186_base = df_GJ186_base.values[(d1-1-1+d2-1):,:]

	# print("GJ186_base.shape=", GJ186_base.shape)  # 调试信息

	GJ186 = (GJ186_base[-1,:] + GJ186_base[0,:]) / 2   # GJ186 已经是 Array的形式
	GJ186 = 1/GJ186  # 测试后无效
	alpha = GJ186 * self.Universe_one.iloc[i,:]
	return alpha






def GJ_alpha187_totest(self):
	# #############################################################
	# GJ#187 = SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
	# 含义：开盘价 <=前一日开盘价 --> 0
	#      开盘价 > 前一日开盘价 --> 取 (最高价-开盘价) 和 (开盘价-前一日开盘价) 的较大值； 
	#      然后把上述取值在过去20天内做累加 
	#    
	# --------------------------------------------
	# 策略方向：买入
	# 主要买入：
	# 主要卖出：
	# --------------------------------------------
	# 评价: Momentum策略 幅度大小取决于 当日开涨的幅度 和 当日最高价涨幅 中的较大值
	# --------------------------------------------
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	DELAY = self.DELAY
	d = 20+1 

	GJ187_highadj = self.HighPrice[di-DELAY-(d)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d)+1:di-DELAY+1,:]
	GJ187_openadj = self.OpenPrice[di-DELAY-(d)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d)+1:di-DELAY+1,:]

	GJ187_cond = np.diff(GJ187_openadj, axis=0)<=0 
	GJ187_value = (GJ187_highadj[1:,:]>=GJ187_openadj[1:,:]) * (GJ187_highadj[1:,:]-GJ187_openadj[1:,:]) + (GJ187_highadj[1:,:]<GJ187_openadj[1:,:]) * np.diff(GJ187_openadj, axis=0)

	GJ187 =np.nanmean(np.where(GJ187_cond, np.zeros((GJ187_openadj[1:,:].shape)),GJ187_value), axis=0, keepdims=True)

	alpha = GJ187[0,:] * self.Universe_one.iloc[i,:]

	return alpha 







def GJ_alpha188_totest(self):
	# #############################################################
	# GJ#188 = ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
	# 含义：高低价差的 相对于过去11天移动平均值的 增长率 
	# 
	# 策略方向：买卖
	# 主要买入：高低价差放大的股票： 1.波动大，2.可能要涨
	#---------------------------------------------
	# 评价：是否是在强势的市场中才有效；这个似乎是只在做volatility？还是过去历史上大多数价差放大意味着上涨/下跌？逻辑是？
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	d = 11 
	
	GJ188_highadj = self.HighPrice[di-DELAY-(d)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d)+1:di-DELAY+1,:] 
	GJ188_lowadj = self.LowPrice[di-DELAY-(d)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d)+1:di-DELAY+1,:] 

	GJ188 = ((GJ188_highadj - GJ188_highadj)[d-1:d,:] / self._SMA((GJ188_highadj-GJ188_lowadj),11,2) ) -1 
	alpha = GJ188[0,:] * self.Universe_one.iloc[i,:]
	return alpha 
	
	
	
	

def GJ_alpha189_totest(self):
	# #############################################################
	# GJ#189 = MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
	#
	# 策略方向：买入
	# 主要买入：买入收盘价与过去6天均价之差的绝对值 越大 
	#---------------------------------------------
	# 评价
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	d1=6
	d2=6
	
	GJ189_closeadj = self.ClosePrice[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] 

	df_GJ189_closeadj = pd.DataFrame(GJ189_closeadj)

	GJ189 = np.nanmean(np.abs((GJ189_closeadj[-6:,:]-(df_GJ189_closeadj.rolling(6,2).mean().values[-6:,:]))), axis=0, keepdims = True)

	alpha = GJ189[0,:] * self.Universe_one.iloc[i,:]
	return alpha 



def GJ_alpha190_totest(self):
	# #############################################################
	# GJ_alpha# : LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
	# LOG(
	#        (COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)
	#       *(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))
	#     /  
	#        ((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))
	#       *(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1)))
	#    )
	# 含义：相对于过去20天平均return的涨跌强弱 
	# 
	# condition: return > 过去20天的几何平均return 
	# 
	# 策略方向：买入
	# 主要买入：
	# 主要卖出：
	#---------------------------------------------
	# 评价
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	DELAY = self.DELAY

	d1 = 20 

	GJ190_closeadj = self.ClosePrice[di-DELAY-(d1+d1-1)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d1+d1-1)+1:di-DELAY+1,:]

	np.diff(GJ190_closeadj,)

	GJ190_ret = np.diff(GJ190_closeadj, axis=0)/GJ190_closeadj[:-1,:] - 1
	GJ190_ret_20 = np.diff(GJ190_closeadj, n=19, axis=0)/GJ190_closeadj[:-19] - 1
	GJ190_cond_expr = GJ190_ret[-20:] - GJ190_ret_20  # 逻辑判断的表达式 
	GJ190_numerator =  np.nansum((np.power(GJ190_cond_expr,2) * (GJ190_cond_expr<0)), axis=0, keepdims=True)  # 分子部分 
	GJ190_denominator = np.nansum((np.power(GJ190_cond_expr,2) * (GJ190_cond_expr>0)), axis=0, keepdims=True) # 分母部分

	GJ190 = np.log(GJ190_numerator / GJ190_denominator)

	alpha = GJ190[0,:] * self.Universe_one.iloc[i,:]
	return alpha 






def GJ_alpha191_totest(self):
	# #############################################################
	# GJ_alpha# : ((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) -CLOSE)
	# 含义
	#
	# 策略方向：
	# 主要买入：
	# 主要卖出：
	#---------------------------------------------
	# 评价
	# --------------------------------------------
	#
	# 有效性趋势：
	# by: XX
	# last modify:
	# #############################################################
	
	d1 = 20 
	d2 = 5 
	
	GJ191_turn = self.Turnover[di-DELAY-(d1+d2-1)+1:di-DELAY+1,:] 
	GJ191_lowadj = self.LowPrice[di-DELAY-(d2)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d2)+1:di-DELAY+1,:] 
	GJ191_high = self.HighPrice[di-DELAY:di-DELAY+1,:]
	GJ191_low = self.LowPrice[di-DELAY:di-DELAY+1,:]
	GJ191_close = self.ClosePrice[di-DELAY:di-DELAY+1]

	GJ191_mean_turn = np.zeros((d2,GJ191_turn.shape[1]))
	for ii in range(d2):
		jj = ii + d1 
		GJ191_mean_turn[ii] = np.nanmean(GJ191_turn[ii:jj], axis=0, keepdims=True)

	GJ191 = self._correlation(GJ191_mean_turn, GJ191_lowadj) + (GJ191_high + GJ191_low)/GJ191_close

	alpha = GJ191[0,:] * self.Universe_one.iloc[i,:]
	return alpha 
