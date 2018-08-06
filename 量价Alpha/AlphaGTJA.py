

import pandas as pd 
import numpy as np 



def GJ_alpha(self):
	# #############################################################
	# GJ_alpha# :
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
	
	这里我做一个改动，看其他地方能不能pull push
	# alpha = GJ[0,:] * self.Universe_one.iloc[i,:]
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






def GJ_alpha187_tovarify(self):
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
	
	d = 20+1 
	
	GJ187_highadj = self.HighPrice[di-DELAY-(d)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d)+1:di-DELAY+1,:]
	GJ187_openadj = self.OpenPrice[di-DELAY-(d)+1:di-DELAY+1,:] * self.adjfactor[di-DELAY-(d)+1:di-DELAY+1,:]
	
	GJ187_cond = np.diff(GJ187_openadj)>0 
	GJ187 =np.nanmean(np.where(GJ187_cond, np.nanmax(GJ187_highadj-GJ187_openadj)[1:], np.diff(GJ187_openadj)), axis=0, keepdims=True)
	
	alpha = GJ187[0,:] * self.Universe_one.iloc[i,:]
	return alpha 





def GJ_alpha189_tovarify(self):
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



def GJ_alpha190_tba(self):
	# #############################################################
	# GJ_alpha# : LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
	# LOG(
	#      (COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)
	#     *(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))
	#     /((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1)))
	#    )
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
	
	
	GJ190 
	alpha = GJ190[0,:] * self.Universe_one.iloc[i,:]
	return alpha 
