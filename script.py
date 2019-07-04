from comet_ml import Experiment
import pandas as pd
from tgan.model import TGANModel
import argparse

def get_data(ds, drop=None, n_unique=20, sep=';', suffix='cat'):
    d = pd.read_csv(f'../data/{ds}/{ds}_{suffix}.csv', sep=sep)
    if drop is not None:
        d = d.drop(drop, axis=1)
        
    continuous_columns = []
    for col in d._get_numeric_data().columns:
        if len(d[col].unique()) > n_unique:
            continuous_columns.append(d.columns.get_loc(col))
    return d, continuous_columns


parser = argparse.ArgumentParser(description='Evaluate data synthesizers')
parser.add_argument('--dataset', nargs='*', help='Which dataset to choose. Options are berka, creditcard and ticket', default=['berka', 'census', 'creditcard'])

args = parser.parse_args()
datasets = args.dataset


for ds in datasets:
    
    if ds == 'berka':
        d, continuous_columns = get_data(ds, drop=['trans_bank_partner', 'trans_account_partner'])
    elif ds == 'census':
        d, continuous_columns = get_data(ds, sep=',')    
    elif ds == 'creditcard':
        d, continuous_columns = get_data(ds, sep=',', suffix='num')

    project_name = "tgan-skip-connections"
    experiment = Experiment(api_key="49HGMPyIKjokHwg2pVOKWTG67",
                            project_name=project_name, workspace="baukebrenninkmeijer")

    tgan = TGANModel(continuous_columns, 
                     restore_session=False,  
                     max_epoch=100, 
                     steps_per_epoch=5000, 
                     batch_size=200,
                     experiment=experiment,
                     num_gen_rnn=50,
                     num_gen_feature=64)
    tgan.fit(d)

    model_path = f'model/{ds}_{project_name}'

    num_samples = 100000
    new_samples = tgan.sample(num_samples)

    p = new_samples.copy()
    if ds == 'berka' or ds == 'census':
        p[p._get_numeric_data().columns] = p[p._get_numeric_data().columns].astype('int')
    if ds == 'creditcard':
        p[['time', 'class']] = p[['time', 'class']].astype('int')
    p.to_csv(f'samples/{ds}_sample_{project_name}_2layerskip.csv', index=False)
    experiment.log_dataset_info(name=ds)
    experiment.end()

    tgan.save(model_path, force=True)
