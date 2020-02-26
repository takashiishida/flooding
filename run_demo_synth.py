import sys, os, subprocess
import numpy as np
import mlflow

from synthetic.get_results_synth import get_data
from synthetic.make_graph_synth import make_graph


def main():
    dataset = 'sinusoid2d'
    params_dict = {
        'setting1': {
            'optimizer': 'sgd',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'fl_arr': np.r_[0.0, 0.26, 0.27, 0.28],
        },            
        'setting2': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'momentum': 0.0, # not used
            'fl_arr': np.r_[0.0, 0.24, 0.25, 0.26],
        }                    
    }    
    d = 2
    epochs = 200
    label_noise = 0.1
    
    setting = 'setting1'
    setting_dict = params_dict[setting]
    optimizer = setting_dict['optimizer']
    learning_rate = setting_dict['learning_rate']
    momentum = setting_dict['momentum']
    
    model = 'mlp_model'    
    fl_arr = setting_dict['fl_arr']
    
    
    for fl in fl_arr:
        cmd = 'mlflow run . -e synthetic --no-conda'.split(' ')
        tag = '-P labels=synth,simple'.split(' ')
        opt = '-P epochs={e} -P flood_level={fl} -P label_noise={ln} -P optimizer={opt} -P model={model}'\
            ' -P dataset={ds} -P dimension={d} -P learning_rate={lr} -P momentum={mm}'.format(
            e=epochs, fl=fl, ln=label_noise, opt=optimizer, model=model, ds=dataset, d=d, lr=learning_rate, mm=momentum).split(' ')
        cmd += tag + opt
        subprocess.run(cmd)
    
    
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
    sys.path.append(os.getcwd())
    
    tracking_uri_name = 'mlflow_synth.db'    
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///' + tracking_uri_name
    if os.path.exists(tracking_uri_name):
        os.remove(tracking_uri_name)        
    
    main()