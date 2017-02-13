import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import seaborn

import outlier

# Plot hourly data showing trend and outliers using IQR
def plot_stations_hourly(df, stations, night = "Friday", hours= ["7PM", "8PM", "9PM", "10PM", "11PM", "12AM"], find="IQR", title='Hourly Exit Traffic Volumes'):
    
    fig, grid = plt.subplots(nrows=len(hours), ncols=len(stations), sharex=False, sharey=True, squeeze=True, figsize=( 8*len(stations), 2.5*len(hours)))
    for hour, axes in zip(hours, grid):
        for station, ax in zip(stations, axes):
            ts = df.query('Station == @station and Night == @night and Hour == @hour')
            outlier.plot(ts.Exit, outliers=True, interval=False, find=find, ax=ax).set_ylim((0,None))
            ax.axvline(datetime(day=24, month=2, year=2014), linestyle='dotted', linewidth=2, color="grey")
            ax.set_title(station + ": Saturday " + hour, fontsize=16)
#            ax.legend(loc="upper left")

    fig.suptitle(title, fontsize= 22, y=1.02)
    fig.tight_layout()
    
def plot_stations_daily(df, stations, find="IQR", title='Daily Exit Traffic'):
    
    nights = ['Friday', 'Saturday']
    fig, grid = plt.subplots(nrows=len(stations), ncols=len(nights), sharex=False, sharey=True, squeeze=True, figsize=( 8*len(nights), 2.5*len(stations)))
    for station, axes in zip(stations, grid):
        for night , ax in zip(nights, axes):
            ts = df.query('Station == @station and Night == @night')
            outlier.plot(ts.Exit, outliers=True, interval=False, find=find, ax=ax).set_ylim((0,None))
            ax.axvline(datetime(day=24, month=2, year=2014), linestyle='dotted', linewidth=2, color="grey")
            ax.set_title(station + " " + night + " night", fontsize=16)

    fig.suptitle(title, fontsize= 22, y=1.02)
    fig.tight_layout()
    
import statsmodels.tsa.api as smt
import statsmodels.api as sm

def plot_autocorrelation(ts, title="Autocorrelation"):
    
    fig =  plt.figure(figsize=(16,4))
    fig.suptitle(title, fontsize=12, y=1.05)

    ts.plot(ax = plt.subplot2grid((2,3), (0,0), colspan=3)).set_title('Time series residues', fontsize=10)
    smt.graphics.plot_acf(ts, ax=plt.subplot2grid((2,3), (1,0)))
    smt.graphics.plot_pacf(ts,  ax=plt.subplot2grid((2,3), (1,1)))
    sm.qqplot(ts,  line='s',  ax= plt.subplot2grid((2,3), (1,2)))

    fig.tight_layout()

def plot_ac(df, station, night, hour, col='Exit'):
    
    ts = (df.query('Station == @station and Night == @night and Hour == @hour').
          groupby(level='Date')['Exit', 'Entry'].sum())

   
    plot_autocorrelation(outlier.residue(ts[col]), "{:} {:} Night {:} {:} Traffic".format(station, night, hour,  col))
    
def set_titles(g):
    '''Set titles correctly to row and col names'''
    # https://github.com/mwaskom/seaborn/issues/440
    # https://github.com/mwaskom/seaborn/issues/706
    for ax in g.axes.flat:
        plt.setp(ax.texts, text="")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    return g


# Support utilities
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from functools import partial
import matplotlib.transforms as transforms
from matplotlib.ticker import MaxNLocator
import datetime as dt

def find_changes(series, truncate=-np.inf):

    data = series.to_frame()
    Q, P, Pcp = offcd.offline_changepoint_detection(data, partial(offcd.const_prior, l=(len(data)+1)), offcd.gaussian_obs_log_likelihood, truncate=truncate)
    data['Change'] =  np.append([0], np.exp(Pcp).sum(0))
    
    return data.Change

def plot_source(df, station, night, col='Exit'):
    ax = (df[df.Hour >= "7PM"].
              query('Station == @station and Night == @night').
              reset_index().
              pivot_table(index='Date', columns='Source', values=col,aggfunc='sum').
              plot(figsize=(12, 2)))
    mark('24-Feb-2014', [ax], text="Lockout", color='grey')
    ax.set_title("{:} {:} Night {:} Traffic".format(station, night, col), fontsize=12)
    
    return ax
    
def plot_change(ts, change):
    
    assert len(ts) != 0, "No data selected"
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,
                               gridspec_kw={'height_ratios': [4, 1]},
                               figsize=(12, 3))     
    ts.plot(ax=axes[0])
    change.plot.area(ax=axes[1])
    axes[1].yaxis.set_major_locator(MaxNLocator(3))

    mark('24-Feb-2014', axes, text="Lockout", color='grey')

    return (axes)

def plot_bcp(df, station, night, col='Exit'):
    '''Convenience function'''
    ts = df.query('Station == @station and Night == @night')

    axes = plot_change(ts[col], ts['Change'+col])
    axes[0].set_title("{:} {:} Night {:} Traffic".format(station, night, col), fontsize=12)
    return ts, axes

def mark(dates, axes, text="{}", color="red", linewidth=1):
          
    if not isinstance(dates, list):
        dates = list([dates])
        
    trans = transforms.blended_transform_factory(axes[0].transData, axes[0].transAxes)
    for date in dates:
        X = dt.datetime.strptime(date, '%d-%b-%Y').date()

        axes[0].text(X + dt.timedelta(days=7), 0, text.format(date), fontsize=8, transform=trans)
        for ax in axes:
            ax.axvline(date, linestyle='dotted', linewidth=linewidth, color=color)
            
            
def segment(ts, dates, axes):

    if not isinstance(dates, list):
        dates = list([dates])
    
    mark(dates, axes)
    
    dates = list(map(lambda d: dt.datetime.strptime(d, '%d-%b-%Y'), dates))
    dates = sorted(dates + [ts.index.min(), ts.index.max()])

    for date1, date2 in zip(dates, dates[1:]):
        outlier.plot(ts[date1 + dt.timedelta(days=7):date2 - dt.timedelta(days=7)], ax=axes[0])

    axes[0].legend(loc='upper left')