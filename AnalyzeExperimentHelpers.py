import os
from os import path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns
import re

def chop(f):
    """
        removes everything after the last "-"
        remove "-train.csv" or "-val.csv"
    """
    i = f.rindex("-")
    return f[:i]
    
def read_metric_frame(fpre, train, metric_dir):
    """
        reads in a specific results file given a file prefix such as "00-LTI-AB-trail0"
    """
    trial_num = int(re.findall(f"trial(\d+)",fpre, re.IGNORECASE)[0])

    if train:
        p = path.join(metric_dir,fpre+"-train.csv")
    else:
        p = path.join(metric_dir,fpre+"-val.csv")
    df = pd.read_csv(p, index_col=[0,1])
    df.index.names = ["state","stat"]
    df.columns = pd.MultiIndex.from_product([[trial_num], df.columns.tolist()], names=["trial","metric"])
    return df

def metric_short_to_long(df):
    """
        simply melts a metrics dataframe
    """
    return df.reset_index().melt(id_vars=["state","stat"])

def _get_experiment_id(TRAIN_CONFIGS):
    """
        e.g. (00)
    """
    return TRAIN_CONFIGS.get("lti_file").split("-")[0]

def _metric_file_prefixes(TRAIN_CONFIGS):
    """
        converts file names in a directory into valid prefixes and removes duplicates 
    """
    data_dir = TRAIN_CONFIGS.get("data_dir")
    ID =  _get_experiment_id(TRAIN_CONFIGS) # e.g. (00)-
    mpre = list(set([chop(f) for f in os.listdir(TRAIN_CONFIGS.get("metrics_dir")) if f[0:len(ID)] == ID]))
    return mpre


def read_experiment_metrics(TRAIN_CONFIGS):
    """
        reads in metrics for all trials, compiling the results into a dataframe for training metrics
        and another for validation metrics.
    """
    metric_dir = TRAIN_CONFIGS.get("metrics_dir")
    mpre = _metric_file_prefixes(TRAIN_CONFIGS)

    mtrain = pd.concat([read_metric_frame(f, train=True, metric_dir=metric_dir) for f in mpre],axis=1)
    mval   = pd.concat([read_metric_frame(f, train=False, metric_dir=metric_dir) for f in mpre],axis=1)

    mtrain = mtrain.sort_index(axis=1,level=0)
    mval   = mval.sort_index(axis=1,level=0)

    return mtrain, mval

def _plot_experiment(df, axes, metric_name, isTrain):
    """
        plots metrics onto two axis. The first axis receives the aggregated state metrics
        The second axis receives the unaggregated state metrics
    """
    # colors: https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
    ldf = metric_short_to_long(df)
    plotted = "Train" if isTrain else "Val"
    m = ldf.query("stat == 'mse' and metric == @metric_name")[["trial","state","value"]].rename({"value":"mse"},axis=1)
    # aggregated
    ax = sns.barplot(x="trial", y="mse", data=m, palette=[u'#1f77b4'], ci="sd", ax=axes[0])
    ax.set_ylabel("MSE (log)")
    ax.set_yscale("log")
    ax.set_title(f"Aggregated State Errors ({plotted})")
    ax.set_xlabel("Trial Number")

    # individual state plots
    ax = sns.barplot(x="trial", y="mse", hue="state",data=m, ci="sd", ax=axes[1])
    ax.set_ylabel("MSE (log)")
    ax.set_yscale("log")
    ax.set_title(f"State Error by Trial ({plotted})")
    ax.set_xlabel("Trial Number")


def _plot_train_test_experiment(mtrain, mval, metric_name, isState):
    """
        plots a figure with 4 plots. 2 plots are for training results for the metric of interest
        the other 2 plots are for validation results for the metric of interest

        metric_name: valid values are {"Delta Percent", "Relative Spacing", "Value"}
    """
    # axes
    f, axes = plt.subplots(2,2,figsize=(12,10))
    ltrain = _plot_experiment(mtrain, axes[:,0], metric_name, isTrain=True)
    lval = _plot_experiment(mval, axes[:,1], metric_name, isTrain=False)
    # title
    target = "State" if isState else "Output"
    f.suptitle(f"{target} Errors")
    f.tight_layout()
    return f, axes

### This one plots the experiment res ### 
def plot_experiment_results(TRAIN_CONFIGS, metric, save):
    """
        This will plot all the trials for a given "experiment" as bar graphs.

        metric: valid values are {"Delta Percent", "Relative Spacing", "Value"}
    """
    name_map = {
        "Delta Percent" : "delta_percent",
        "Relative Spacing" : "rel_space",
        "Value": "value"
    }

    isState = TRAIN_CONFIGS.get("target") == "states"
    mtrain, mval = read_experiment_metrics(TRAIN_CONFIGS)
    f, axes = _plot_train_test_experiment(mtrain, mval, "Relative Spacing", isState)
    if save:
        ID = _get_experiment_id(TRAIN_CONFIGS)
        fname = f"{ID}-results-"+name_map[metric]
        p = path.join(TRAIN_CONFIGS.get("fig_dir"), fname)
        f.savefig(fname=p,facecolor="white",edgecolor='none')
    return f, axes