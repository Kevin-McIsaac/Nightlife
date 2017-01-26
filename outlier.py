# A simple module for visualising, identifing and replace outliers
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm as Gaussian
from statsmodels.tsa.api import detrend as sm_detrend
from statsmodels.formula.api import ols 
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from numpy import issubdtype, number


def isoutlier(ts, N = 6, find="IQR", detrend=True, order=1):
    '''Find outliers in a time series.
    
       ts:      A DataFrame or Series with a timeseries index
                with all numeric columns
       
       find:    {'MAD', 'IQR, 'ZScore'}
                Method of finding outliers
              
       N:       if |MAD or ZScore| > sd observation is an outlier
       
       detrend: remove the timeseries trend before finding ourliers
                see statsmodels.tsa.tsatools.detrend
       
       order:   polynomial order of the trend, zero is constant, one is linear trend, two is quadratic
       
       returns: Same shape Series or DataFrame of boolean values '''
    
    _checkTimeSeries(ts)  # basic timeseries sanity checks that raise ValueError
  
    if detrend:
        if type(ts) == pd.Series: 
            #Applying detrend to a Series applied it element by element.
            ts = pd.Series(sm_detrend(ts, order=order), ts.index)
        else:
            ts = ts.apply(sm_detrend, order=order)
            
    if find == "MAD":
        outliers = abs(ts - ts.median()) > ts.mad() * (N / Gaussian.ppf(3/4.)) # usual equation is reordered for efficency
        
    elif find == "ZScore":
        outliers = (ts - ts.mean()) > ts.std() * N # usual equation is reordered for efficency
        
    elif find == "IQR": # Note uses a fixed value rather than N
        q = ts.quantile([0.25,.75]).T
        IQR = q[0.75] - q[0.25]
        outliers = (ts < (q[0.25] - 1.5*IQR)) | (ts > (q[0.75] + 1.5*IQR))
        
    else:
        raise ValueError('find must be one of "MAD", "ZScore" or "IQR"') 

    return outliers

def plotts(ts, trend=True, interval=False, outliers=False,  ax=None,  **kwargs):
    '''Plot a timeseries optionally overlaying trend, 
       2 standard deviation interval and outliers
    
    ts: A DataFrame or Series with a timeseries index with all numeric columns
    
   
    trend:     overlay trend linear?
    
    interval:  overlay a 2 standard deviation interval?
    
    outliers:  overlay outliers?
    
    kwargs:    aguments passed to isoutler
    
    ax:        axes to draw on (optional)
    
    returns: axes'''
    
    _checkTimeSeries(ts)  # basic timeseries sanity checks that raise ValueError
    
    if not ax:
        ax = plt.subplot(111)

    # ols won't accept a date so create time in seconds from first date as the independant variable  
    if isinstance(ts, pd.Series):
        df = (ts).to_frame() # Unify handeling of Series and DataFrame
    else:
        df = ts.copy()
    
    cols = df.columns
    df['__Seconds'] = (ts.index - ts.index.min()).astype('timedelta64[s]') 
    for col in cols:
        res = ols(formula=col+ ' ~ __Seconds', data=df).fit()

        
        if trend:
            label = "{} (r^2 = {:2.2})".format(col, res.rsquared)
            ax.plot(res.fittedvalues, '--g')
        else:
            label = col 
              
        if interval:
            prstd, iv_l, iv_u = wls_prediction_std(res)
            ax.fill_between(iv_l.index, iv_l, iv_u, color='#888888', alpha=0.25) 
        if outliers: 
            ax.plot(df[col][isoutlier(df[col], **kwargs)], 'r*', label="outliers")           
        ax.plot(df[col], label=label)

    return ax
            
def replace(ts, find="MAD", detrend=True, order=1, how='NaN', **kwargs):
        ''' Replace outliers in ts with NaN or interpolated values. 
            method: {}
            . 
        '''
        ts = ts.where(~isoutlier(ts, find=find, detrend=detrend, order=order))
        if how != 'NaN':
            ts = ts.interpolate(how=how, **kwargs)
            
        return ts

    
# ============ utility functions =====================

def _checkTimeSeries(ts):
    '''basic sanaity checks with clear messages'''
    
    if not (isinstance(ts, pd.Series) or isinstance(ts, pd.DataFrame) ):
        raise ValueError('Timeseries must be a pandas Series or DataFrame') 
    
    if not _isnumeric(ts):
        raise ValueError('Timeseries must have all numberic columns')
        
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError('Timeseries index must be a DatetimeIndex') 
    
# this would be better implemneted as methods on Series and DataFrame
def _isnumeric(df):
    '''Check that all colums are numeric types'''
   
    return ((isinstance(df, pd.Series)    and issubdtype(df, number)) or 
            (isinstance(df, pd.DataFrame) and  all([issubdtype(t, number) for t in df.dtypes])))
