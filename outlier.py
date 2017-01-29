# A simple module for visualising, identifing and replace outliers
import pandas as pd
from matplotlib.pyplot import gca

from scipy.stats import norm as Gaussian
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np 


def isoutlier(ts, find="IQR", N=2, detrend=True):
    '''
    Find outliers in a time series.
    Parameters
    ----------
    ts : A DataFrame or Series with a timeseries index
        with all numeric columns
       
    find : {'MAD', 'IQR, 'ZScore'}
        Method of finding outliers
              
    N : if |MAD or ZScore| > N observation is an outlier
       
    detrend : remove the timeseries trend before finding ourliers
        see statsmodels.tsa.tsatools.detrend
       
    Returns
    -------
    Series or DataFrame of boolean values of the same shape 
    '''
      
    if detrend:
        ts = residue(ts) 
            
    if find == "MAD":
        outliers = abs(ts - ts.median()) > ts.mad() * (N / Gaussian.ppf(3/4.)) # reordered to avoid vector division
        
    elif find == "ZScore":
        outliers = (ts - ts.mean()) > ts.std() * N # reordered to avoid vector division
        
    elif find == "IQR": # Note uses a fixed value rather than N
        q = ts.quantile([0.25,.75]).T
        IQR = q[0.75] - q[0.25]
        outliers = (ts < (q[0.25] - 1.5*IQR)) | (ts > (q[0.75] + 1.5*IQR))
        
    else:
        raise ValueError('find must be one of "MAD", "ZScore" or "IQR"') 

    return outliers

def trend(ts):
    '''Return the line of best fit for numeric columns'''
    
    df = ts.copy() if not isinstance(ts, pd.Series) else ts.to_frame() # Unify handeling of Series and DataFrame
    
    cols =  df.select_dtypes(include=[np.number]).columns # Only work on numeric columns
    df['__Seconds'] = (ts.index - ts.index.min()).astype('timedelta64[s]') 

    for col in cols:
        df[col] = smf.ols(formula=col+ ' ~ __Seconds', data=df).fit().fittedvalues
    
    return df[cols] if not isinstance(ts, pd.Series) else df[cols[0]] # return numeric columns as dataframe or series

def residue(ts):
    '''Return the residue from the line of best fit for numeric columns'''
    
    df = ts.copy() if not isinstance(ts, pd.Series) else ts.to_frame() # Unify handeling of Series and DataFrame
    
    cols =  df.select_dtypes(include=[np.number]).columns # Only work on numeric columns
    df['__Seconds'] = (ts.index - ts.index.min()).astype('timedelta64[s]') 

    for col in cols:
        df[col] = smf.ols(formula=col+ ' ~ __Seconds', data=df).fit().resid

    return df[cols]if not isinstance(ts, pd.Series) else df[cols[0]] # return numeric columns as dataframe or series


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
    
    # ols won't accept a date so create time in seconds from first date as the independant variable  
    if isinstance(ts, pd.Series):
        df = (ts).to_frame() # Unify handeling of Series and DataFrame
    else:
        df = ts.copy()
    
    cols =  df.select_dtypes(include=[np.number]).columns
    df['__Seconds'] = (ts.index - ts.index.min()).astype('timedelta64[s]') 
    for col in cols:

        res = smf.ols(formula=col+ ' ~ __Seconds', data=df).fit()

        # Plot this first to get the better pandas timeseries drawing of dates on x axis
        df[col].plot(ax=ax, label="{} (r^2 = {:2.2})".format(col, res.rsquared) if trend else col)              
        
        if trend:
            res.fittedvalues.plot(ax=ax, style='--g', label="")
        if interval:
            prstd, iv_l, iv_u = wls_prediction_std(res)
            ax.fill_between(iv_l.index, iv_l, iv_u, color='#888888', alpha=0.25) 
        if outliers:
            df_outliers = df[col][isoutlier(df[col], **kwargs)]
            if len(df_outliers) > 0:
                df_outliers.plot( ax=ax, style='r*', label="")           

    return ax
            
def replace(ts, find="IQR", detrend=True, N = 2, how='NaN', **kwargs):
        ''' 
        Replace outliers in ts with NaN or interpolated values. 
        Parameters
        ----------
        ts : A DataFrame or Series with a timeseries index
            with all numeric columns

        find : {'MAD', 'IQR, 'ZScore'}
            Method of finding outliers

        N : if |MAD or ZScore| > N observation is an outlier

        detrend : remove the timeseries trend before finding ourliers
            see statsmodels.tsa.tsatools.detrend

        Returns
        -------
        Series or DataFrame of the same shape 
        
        '''
        ts = ts.where(~isoutlier(ts, find=find, N=N, detrend=detrend))
        if how != 'NaN':
            ts = ts.interpolate(how=how, **kwargs)
            
        return ts