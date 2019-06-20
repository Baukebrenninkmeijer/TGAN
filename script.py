from comet_ml import Experiment
import pandas as pd
from tgan.model import TGANModel

def get_data(ds, drop=None, n_unique=20):
    d = pd.read_csv(f'../data/{ds}/{ds}_cat.csv', sep=';')
    if drop is not None:
        d = d.drop(drop, axis=1)
        
    continuous_columns = []
    for col in d._get_numeric_data().columns:
        if len(d[col].unique()) > n_unique:
            continuous_columns.append(d.columns.get_loc(col))
    return d, continuous_columns

ds = 'Ticket'
# d, continuous_columns = get_data(ds, drop=['trans_bank_partner', 'trans_account_partner'])
d, continuous_columns = get_data(ds)

project_name = "tgan-wgan-gp"
experiment = Experiment(api_key="49HGMPyIKjokHwg2pVOKWTG67",
                        project_name=project_name, workspace="baukebrenninkmeijer")

tgan = TGANModel(continuous_columns, 
                 restore_session=False,  
                 max_epoch=100, 
                 steps_per_epoch=1000, 
                 batch_size=500,
                 experiment=experiment,
                 num_gen_rnn=50,
                 num_gen_feature=64
                )
                
tgan.fit(d)

model_path = f'model/{ds}_{project_name}'

num_samples = 100000
new_samples = tgan.sample(num_samples)

p = new_samples.copy()
p[p._get_numeric_data().columns] = p[p._get_numeric_data().columns].astype('int')
p.to_csv(f'samples/{ds}_sample_{project_name}_2layerskip.csv', index=False)
experiment.log_asset_data(p, file_name=f'sample_{ds}_{project_name}_{len(p)}', overwrite=False)
experiment.log_dataset_info(name=ds)
experiment.end()

tgan.save(model_path, force=True)
