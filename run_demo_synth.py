import sys, os, subprocess
import numpy as np
import mlflow

from synthetic.get_results_synth import get_data
from synthetic.make_graph_synth import make_graph


def main():
    uri_mlproject = os.getcwd()  # Assumes MLproject is preset in the current directory.
    mlflow.set_tracking_uri('sqlite:///mlflow_synth.db')
    basic_setting = {
        'labels': 'synth,simple',
        'epochs': 200,
        'label_noise': 0.1,
        'model': 'mlp_model',
        'dataset': 'sinusoid2d',
        'dimension': 2,
    }
    params_choices = {
        'setting1': {
            'optimizer': 'sgd',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'fl_arr': np.r_[0.0, 0.26, 0.27, 0.28],
        },
        'setting2': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'fl_arr': np.r_[0.0, 0.24, 0.25, 0.26],
        }
    }

    params = dict(params_choices['setting1'], **basic_setting)  # Concatenate the configurations.
    fl_arr = params.pop('fl_arr')  # Drop it; The python script wouldn't accept it as an argument.

    with mlflow.start_run() as run:
        for fl in fl_arr:
            tmp_params = params.copy()
            tmp_params['flood_level'] = fl
            mlflow.run(uri=uri_mlproject, entry_point='synthetic', parameters=tmp_params, use_conda=False)
    
    query = 'tags."synth" = "True" and tags."simple" = "True" and attribute.status = "FINISHED"'
    df = get_data(query)
    
    acc_wo_fl = df[df['fl'].values == '0.0']
    
    chosen_fl_idx = df['vaAcc'].values.argmax()
    chosen_fl = df.iloc[chosen_fl_idx]['fl']
    acc_w_fl = df.iloc[[chosen_fl_idx]]    
    
    print('Test Acc. {:.3f}, Train Acc. {:.3f}: without flooding'.format(acc_wo_fl['teAcc'].values[0], acc_wo_fl['trAcc'].values[0]))
    print('Test Acc. {:.3f}, Train Acc. {:.3f}: with flooding {}'.format(acc_w_fl['teAcc'].values[0], acc_w_fl['trAcc'].values[0], chosen_fl))
    
    make_graph(query, chosen_fl)
                

if __name__ == "__main__":    
    main()

