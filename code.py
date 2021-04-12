import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def zsret(file,name,length):
    indret=[]
    pri = pd.read_excel(file,sheet_name=name).iloc[5:,1]
    if len(pri)>length: 
       for i in range(len(pri)-length):
           indret.append(pow(pri.iloc[i+length]/pri.iloc[i],12/length)-1)
    return indret

def retset(length):
    # -------欧美发达市场--------
    ukx = zsret('欧美发达市场.xlsx', '富时100UKX', length)
    ibe = zsret('欧美发达市场.xlsx', '西班牙 IBEX', length)
    dax = zsret('欧美发达市场.xlsx', '德国DAX', length)
    cac = zsret('欧美发达市场.xlsx', '法国CAC', length)
    ind = zsret('欧美发达市场.xlsx', '道琼斯INDU', length)
    spx = zsret('欧美发达市场.xlsx', '标普SPX', length)
    # -------其他新兴市场--------
    mex = zsret('其他新兴市场.xlsx', '墨西哥MEXBOL', length)
    nif = zsret('其他新兴市场.xlsx', '印度NIFTY', length)
    # mer=zsret('%s/其他新兴市场.xlsx'%(cd),'阿根廷MERVAL',length)
    sen = zsret('其他新兴市场.xlsx', '印度SENSEX', length)
    # -------中国市场--------
    sha = zsret('上证+沪深300+中证500.xlsx', '上证', length)
    zho = zsret('上证+沪深300+中证500.xlsx', '中证', length)
    hus = zsret('上证+沪深300+中证500.xlsx', '沪深', length)
    # -------亚太市场--------
    nky = zsret('亚太市场.xlsx', '日经NKY', length)
    kos = zsret('亚太市场.xlsx', '韩国KOSPI', length)
    hsi = zsret('亚太市场.xlsx', '恒生HSI', length)
    as5 = zsret('亚太市场.xlsx', '澳大利亚AS51', length)
    allretset =   spx + sha + hus + nky + zho + hsi + mex + nif +sen+ ukx + ibe + dax + cac + ind + kos + as5
    return allretset

def calvol():
    vol = []
    for i in range(12, 481, 12):
        vol.append(np.std(retset(i)))
    return vol

def smooth_sigma(basic):
    y = np.array(basic)
    x1 =np.arange(25,65)
    def func(x, a, b,c):
        return a+b/(c-x)
    popt, pcov = curve_fit(func, x1, y)
    a = popt[0]
    b = popt[1]
    c = popt[2]
    sigma_sim = func(x1, a, b, c)
    plt.plot(x1, sigma_sim)
    plt.xlabel('t')
    plt.ylabel('σ')
    plt.show()
    return sigma_sim

vol=calvol()
sigma1=vol[::-1]
sigma=smooth_sigma(sigma1)
#%%
def simulate_gamma():
    x = np.array([35,45,55])
    y = np.array([-9,-10,-11])
    x1 =np.arange(25,65)

    def func(x, a, b,c):
        return a*x**2+b*x+c
    popt, pcov = curve_fit(func, x, y)
    a = popt[0]
    b = popt[1]
    c = popt[2]
    gamma = func(x1, a, b, c)
    plt.plot(x1, 1-gamma)
    plt.xlabel('t')
    plt.ylabel('γ')
    plt.scatter(x,1-y)
    plt.show()
    return gamma

def simulatext(*, r, mu, sigma, gamma2, x0,pi,T,pattern):
    alphahat = np.zeros(T)
    for i in range(T):
        alphahat[i] = (mu[i]-r[i]) / ((1 - gamma2[i]) * pow(sigma[i],2))
    xset = np.zeros(T)
    gset = np.zeros(T)
    integrate = np.zeros(T)
    integrate_gross = 0
    
    for i in range(T):
        if i == 0:
            integrate[i]=0
        else:    
            integratei=np.exp(-r[i]*i)*pi[i]
            integrate[i]=integrate[i-1]+integratei
            integrate_gross = integrate_gross + integratei
        
    for i in range(T):
        if i == 0:
           gset[i] = alphahat[i]*(1+(integrate_gross-integrate[i])/x0)
           if gset[i] > 1:
              gset[i] = 1
           xset[i] = x0 + mu[i]*gset[i]*x0 + (1-gset[i])*r[i]*x0 + pi[i]   
        else:
           gset[i] = alphahat[i]*(1+(integrate_gross-integrate[i])/xset[i-1])
           if gset[i] > 1:
              gset[i] = 1
           xset[i] = xset[i-1] + mu[i]*gset[i]*xset[i-1] + r[i]*(1-gset[i])*xset[i-1] + pi[i]
           
    days = np.arange(25,65)
    days2 = list(days)
    days2.insert(22,46.5)
    #plt.plot(days2,gset[10:40])
    upgset=[]
    downgset=[]
    for i in range(40):
        if gset[i]==1:
           upgset.append(1)
           downgset.append(0.85)
        else:
           upgset.append(gset[i]+0.1)
           downgset.append(gset[i]-0.15)
        if upgset[i]>1:
           upgset[i]=1
        if upgset[i]<0:
           upgset[i]=0 
        if downgset[i]<0:
           downgset[i]=0 
    plt.plot(days,gset,linestyle=pattern)
    upgset2 = list(upgset)
    upgset2.insert(21,1)
    plt.plot(days2,upgset2)
    plt.plot(days,downgset)
    plt.xlabel('t')
    plt.ylabel('gt')
    return gset,xset,integrate

gamma_2=simulate_gamma()
pi=np.linspace(10000,10000,40)
r=np.linspace(0.04,0.04,40)
mu=np.linspace(0.06,0.06,40)

g,x,integ=simulatext(r=r,mu=mu,sigma=sigma,gamma2=gamma_2,x0=40000, pi=pi,T=40,pattern='-')
plt.show()
