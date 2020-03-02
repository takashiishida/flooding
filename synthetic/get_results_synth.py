import sys
import mlflow
import numpy as np
import pandas as pd
from utils.analysis import MLFlowAnalysis


def get_data(query):
    exps = mlflow.search_runs(experiment_ids=None, filter_string=query)
    
    df_list = []
    for idx, exp in exps.iterrows():
        run_id = exp['run_id']
        
        analysis = MLFlowAnalysis(run_id, tracking_uri=None)
        fl = exp['params.flood_level']
        train_cl_loss, val_cl_loss, test_cl_loss, train_cl_acc, val_cl_acc, test_cl_acc = analysis.prepare_clean_data()
        
        last_cl_ret = {
            'trLss': train_cl_loss[-1:], 'vaLss': val_cl_loss[-1:], 'teLss': test_cl_loss[-1:], 
            'trAcc': train_cl_acc[-1:], 'vaAcc': val_cl_acc[-1:], 'teAcc': test_cl_acc[-1:],
        }
                                
        df_ = pd.DataFrame(data=last_cl_ret)
        df_['fl'] = fl
        
        df_list.append(df_)
    
    return pd.concat(df_list, axis=0)


def get_latest_data(query):
    exps = mlflow.search_runs(experiment_ids=None, filter_string=query)
    run_id = exps['run_id'].values[0]
    analysis = MLFlowAnalysis(run_id, tracking_uri=None)

    train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = analysis.prepare_data()
    train_cl_loss, val_cl_loss, test_cl_loss, train_cl_acc, val_cl_acc, test_cl_acc = analysis.prepare_clean_data()
    return train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, \
               train_cl_loss, val_cl_loss, test_cl_loss, train_cl_acc, val_cl_acc, test_cl_acc
        
    
def get_result(query):
    train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, \
    train_cl_loss, val_cl_loss, test_cl_loss, train_cl_acc, val_cl_acc, test_cl_acc = get_latest_data(query)
                
    last_ret = {
        'trLss': train_loss[-1], 'vaLss': val_loss[-1], 'teLss': test_loss[-1], 
        'trAcc': train_acc[-1], 'vaAcc': val_acc[-1], 'teAcc': test_acc[-1], 
               }
    
    last_cl_ret = {
        'trLss': train_cl_loss[-1], 'vaLss': val_cl_loss[-1], 'teLss': test_cl_loss[-1], 
        'trAcc': train_cl_acc[-1], 'vaAcc': val_cl_acc[-1], 'teAcc': test_cl_acc[-1], 
               }
    
    i_es = np.argmax(val_acc)
    es_ret = {
        'trLss': train_loss[i_es], 'vaLss': val_loss[i_es], 'teLss': test_loss[i_es], 
        'trAcc': train_acc[i_es], 'vaAcc': val_acc[i_es], 'teAcc': test_acc[i_es], 
    }

    i_cl_es = np.argmax(val_cl_acc)
    es_cl_ret = {
        'trLss': train_cl_loss[i_cl_es], 'vaLss': val_cl_loss[i_cl_es], 'teLss': test_cl_loss[i_cl_es], 
        'trAcc': train_cl_acc[i_cl_es], 'vaAcc': val_cl_acc[i_cl_es], 'teAcc': test_cl_acc[i_cl_es], 
    }
    
    return last_ret, last_cl_ret, es_ret, es_cl_ret


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        query = f.readline()
    
    last_ret, last_cl_ret, es_ret, es_cl_ret = get_result(query)
    print(last_cl_ret)