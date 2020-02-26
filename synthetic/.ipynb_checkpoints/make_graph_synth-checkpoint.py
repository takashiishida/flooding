import sys
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.analysis import MLFlowAnalysis


def make_graph(query, chosen_fl):
    query_wo_fl = query + ' and param.flood_level="0.0"'
        
    exps = mlflow.search_runs(experiment_ids=None, filter_string=query)
    for idx, exp in exps.iterrows():
        run_id = exp['run_id']
        analysis = MLFlowAnalysis(run_id, tracking_uri=None)
        tr_loss_wo_fl, va_loss_wo_fl, te_loss_wo_fl, tr_acc_wo_fl, va_acc_wo_fl, te_acc_wo_fl = analysis.prepare_data()
        trcl_loss_wo_fl, vacl_loss_wo_fl, tecl_loss_wo_fl, trcl_acc_wo_fl, vacl_acc_wo_fl, tecl_acc_wo_fl = analysis.prepare_clean_data()        
    
    exps = mlflow.search_runs(experiment_ids=None, filter_string=query_wo_fl)
    run_id = exps['run_id'].iloc[0]        
    analysis = MLFlowAnalysis(run_id, tracking_uri=None)    
    tr_loss_wo_fl, va_loss_wo_fl, te_loss_wo_fl, tr_acc_wo_fl, va_acc_wo_fl, te_acc_wo_fl = analysis.prepare_data()
    trcl_loss_wo_fl, vacl_loss_wo_fl, tecl_loss_wo_fl, trcl_acc_wo_fl, vacl_acc_wo_fl, tecl_acc_wo_fl = analysis.prepare_clean_data()
    
    query_w_fl = query + ' and param.flood_level="{}"'.format(chosen_fl)    
    exps = mlflow.search_runs(experiment_ids=None, filter_string=query_w_fl)
    run_id = exps['run_id'].iloc[0]    
    analysis = MLFlowAnalysis(run_id, tracking_uri=None)    
    tr_loss_w_fl, va_loss_w_fl, te_loss_w_fl, tr_acc_w_fl, va_acc_w_fl, te_acc_w_fl = analysis.prepare_data()
    trcl_loss_w_fl, vacl_loss_w_fl, tecl_loss_w_fl, trcl_acc_w_fl, vacl_acc_w_fl, tecl_acc_w_fl = analysis.prepare_clean_data()
        
    n_epochs = len(trcl_loss_w_fl)
    
    fig, ax = plt.subplots()
    ax.plot(range(n_epochs), tr_loss_wo_fl, label='train w/o flooding')
    ax.plot(range(n_epochs), tecl_loss_wo_fl, label='test w/o flooding')
    ax.plot(range(n_epochs), tr_loss_w_fl, label='train w/ flooding')
    ax.plot(range(n_epochs), tecl_loss_w_fl, label='test w/ flooding')    
    ax.set_title('Loss')
    ax.legend()
    plt.savefig('synth_loss.png')

    
    fig, ax = plt.subplots()
    ax.plot(range(n_epochs), tr_acc_wo_fl, label='train acc. w/o flooding')
    ax.plot(range(n_epochs), tecl_acc_wo_fl, label='test acc. w/o flooding')
    ax.plot(range(n_epochs), tr_acc_w_fl, label='train acc. w/ flooding')
    ax.plot(range(n_epochs), tecl_acc_w_fl, label='test acc. w/ flooding')    
    ax.set_title('Accuracy (1 - Error)')
    ax.legend()        
    plt.savefig('synth_acc.png')        
    