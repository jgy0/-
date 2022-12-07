from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
#1.获取数据
df = pd.read_csv("C:\\Users\Lenovo\Desktop\\1.csv",encoding = 'gbk')
dat=df["data"].tolist()
print(dat)
rng = pd.date_range('2019/11/1',periods = 14, freq = 'D')
print(rng)
dta = pd.Series(dat, index=rng)#时间序列数据
# dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001', '2090'))
print(dta.head())

dta.plot(figsize=(12, 8))
plt.title('dta')
plt.show() #数据图

fig = plt.figure(figsize=(12, 8))
ax1= fig.add_subplot(111)
#差分
diff1 = dta.diff(1)#detalxt = xt-xt-1
# 差分后需要排空，
diff1 = diff1.dropna()
diff1.plot(ax=ax1)
plt.title('diff1')
plt.show()

#3.合适的ARIMA模型
# dta = dta.diff(1)#我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉
# dta = dta.dropna()
dta = diff1
"""
fig = plt.figure(figsize=(12, 8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)
plt.show()
"""
#4.模型优化
arma_mod20 = sm.tsa.ARIMA(dta, (2, 1,1)).fit()# ARIMA(dta,(p,d,q))
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
# arma_mod30 = sm.tsa.ARMA(dta,(0,1)).fit()
# print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
# arma_mod40 = sm.tsa.ARMA(dta,(7,1)).fit()
# print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
# arma_mod50 = sm.tsa.ARMA(dta,(8,0)).fit()
# print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)
#5.模型检验
#5.1模型所产生的残差做自相关图
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_mod20.resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_mod20.resid, lags=40, ax=ax2)
plt.show()
#5.2 D-W检验
#德宾-沃森（Durbin-Watson）检验。德宾-沃森检验,简称D-W检验，是目前检验自相关性最常用的方法，
#但它只使用于检验一阶自相关性。因为自相关系数ρ的值介于-1和1之间，所以 0≤DW≤４。并且DW＝O＝＞ρ＝１
#即存在正自相关性
#DW＝４＜＝＞ρ＝－１　即存在负自相关性
#DW＝２＜＝＞ρ＝０　　即不存在（一阶）自相关性
#因此，当DW值显著的接近于O或４时，则存在自相关性，而接近于２时，则不存在（一阶）自相关性。
#这样只要知道ＤＷ统计量的概率分布，在给定的显著水平下，根据临界值的位置就可以对原假设Ｈ０进行检验。
print(sm.stats.durbin_watson(arma_mod20.resid.values))
#5.3 观察是否符合正态分布
#resid = arma_mod20.resid#残差
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(arma_mod20.resid, line='q', ax=ax, fit=True)
plt.show()
#5.4 Ljung-Box检验
# ARIMA   Ljung-Box检验 -----模型显著性检验，Prod> 0.05，说明该模型适合样本
#Ljung-Box test是对randomness的检验,或者说是对时间序列是否存在滞后相关的一种统计检验。
#对于滞后相关的检验，我们常常采用的方法还包括计算ACF和PCAF并观察其图像，
#但是无论是ACF还是PACF都仅仅考虑是否存在某一特定滞后阶数的相关。
#LB检验则是基于一系列滞后阶数，判断序列总体的相关性或者说随机性是否存在。
#时间序列中一个最基本的模型就是高斯白噪声序列。而对于ARIMA模型，
#其残差被假定为高斯白噪声序列，所以当我们用ARIMA模型去拟合数据时，拟合后我们要对残差的估计序列进行LB检验，
#判断其是否是高斯白噪声，如果不是，那么就说明ARIMA模型也许并不是一个适合样本的模型。Prob<0.05拒绝
r,q,p = sm.tsa.acf(arma_mod20.resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
#3.6 模型预测
predict_sunspots = arma_mod20.predict('2090', '2100', dynamic=True)
print(predict_sunspots)
fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.loc['2001':].plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()
