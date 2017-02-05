# A simple module for visualising, identifing and replace outliers
import pandas as pd
from matplotlib.pyplot import gca

from scipy.stats import norm as Gaussian
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np 


def isoutlier(ts, find="IQR", N=2, detrend=True, level=0):
    '''
    Find outliers in a time series.
    Parameters
    ----------
    ts : A Series with a timeseries index
       
    find : {'MAD', 'IQR, 'ZScore'}
        Method of finding outliers
              
    N : if |MAD or ZScore| > N observation is an outlier
       
    detrend : remove the trend before finding ourliers
       
    level : index level for ts datatime
    
    Returns
    -------
    Series or DataFrame of boolean values of the same shape, discarding 
    non-numeric columns
    '''
    if not isinstance(ts, pd.Series):
        raise ValueError('must be a Series') 

    ts_len = len(ts)
    if detrend:
        ts = residue(ts, level=level) 
            
    if find == "MAD":
        outliers = abs(ts - ts.median()) > ts.mad() * (N / Gaussian.ppf(3/4.)) # reordered to avoid vector division
        
    elif find == "ZScore":
        outliers = (ts - ts.mean()) > ts.std() * N # reordered to avoid vector division
        
    elif find == "IQR": # Note uses a fixed value rather than N
        q = ts.quantile([0.25,.75])
        IQR = q[0.75] - q[0.25]
        outliers = (ts < (q[0.25] - 1.5*IQR)) | (ts > (q[0.75] + 1.5*IQR))
        
    else:
        raise ValueError('find must be one of "MAD", "ZScore" or "IQR"') 

    assert ts_len == len(outliers), "Returned result is incorrect length"
    return outliers

def trend(ts, level=0):
    '''
    Return the line of best fit for the numeric columns
    Parameters
    ----------
    ts : A DataFrame or Series with a timeseries index
    '''
    
    if (~ts.isnull()).sum() < 2 : #need at least two points
        return ts
    else:
        return ols_ts(ts, level=level).fit().fittedvalues

def residue(ts, level=0):
    '''
    Return the residue from the line of best fit for numeric columns
       Parameters
    ----------
    ts : A DataFrame or Series with a timeseries index
        with all numeric columns
    '''
    if (~ts.isnull()).sum() < 2 : #need at least two points
        return ts
    else:
        return ts.where(ts.isnull(), ols_ts(ts, level=level).fit().resid) #Necessary to deal with NaN values in ts

def ols_ts(ts, level=0):

    # TODO use frequency of the series
    # ols does not work on datetime, need to create a dependant variable
    df = ts.to_frame() 
    df['__X'] = ((df.index.get_level_values(level) -df.index.get_level_values(level).min()).
                        astype('timedelta64[s]')) # Ok for most timeseries

    return smf.ols(formula='df[[0]] ~ __X', data=df)


# TODO: Consider non-linear fits
def plot(ts, trend=True, interval=False, outliers=False,  ax=None,  **kwargs):
    '''
    Plot a timeseries with optionl trend, 2 standard deviation interval and outliers
    Parameters
    ----------
    ts : A DataFrame or Series with a timeseries index with all numeric columns
   
    trend : overlay trend linear?
    
    interval : overlay a 2 standard deviation interval?
    
    outliers : overlay outliers?
    
    kwargs : aguments passed to isoutler
    
    ax : axes to draw on (optional)
    
    Returns
    -------
    axes object
    '''
    
    if not ax:
        ax = gca()

    if not isinstance(ts, pd.Series):
        (ts.select_dtypes(include=[np.number]).
                   apply(plot, trend=trend, interval=interval, outliers=outliers,  ax=ax,  **kwargs))
        return ax
    
    result = ols_ts(ts).fit()

    # Plot this first to get the better pandas timeseries drawing of dates on x axis
    ts.plot(ax=ax, label="{} (r^2 = {:2.2})".format(ts.name, result.rsquared) if trend else ts.name)              

    if trend:
        result.fittedvalues.plot(ax=ax, style='--g', label="")
    if interval:
        prstd, iv_l, iv_u = wls_prediction_std(result)
        ax.fill_between(iv_l.index, iv_l, iv_u, color='#888888', alpha=0.25) 
    if outliers:
        df_outliers = ts[isoutlier(ts, **kwargs)]
        if len(df_outliers) > 0:
            df_outliers.plot( ax=ax, style='r*', label="")           

    return ax
            
def replace(ts, find="IQR", detrend=True, N = 2, how='NaN', **kwargs):
        ''' 
        Replace outliers in ts with NaN or interpolated values. 
        Parameters
        ----------
        ts : A DataFrame or Series with a timeseries index.

        find : {'MAD', 'IQR, 'ZScore'}
            Method of finding outliers

        N : if |MAD or ZScore| > N observation is an outlier

        detrend : remove the timeseries trend before finding ourliers

        Returns
        -------
        Series or DataFrame of the same shape non numeric columns are returned unchanged
        
        '''
        if not isinstance(ts, pd.Series):
            return ts.apply(replace,  find=find, detrend=detrend, N = N, how=how, **kwargs)
 
        try:
            ts1 = ts.where(~isoutlier(ts, find=find, N=N, detrend=detrend))
            if how != 'NaN':
                ts1 = ts1.interpolate(how=how, **kwargs)

            return ts1
        except: # ts is probably not a np.numeric
            return ts