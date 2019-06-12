from comet_ml import Experiment
import pandas as pd
from tgan.model import TGANModel

ds = 'berka'
d = pd.read_csv(f'../data/{ds}/{ds}_cat.csv', sep=';')
d = d.drop(['trans_bank_partner', 'trans_account_partner'], axis=1)
continuous_columns = [0, 1, 2, 3, 7]

project_name = "tgan-skip-connections"
experiment = Experiment(api_key="49HGMPyIKjokHwg2pVOKWTG67",
                        project_name=project_name, workspace="baukebrenninkmeijer")

tgan = TGANModel(continuous_columns, 
                 restore_session=False,  
                 max_epoch=100, 
                 steps_per_epoch=2000, 
                 batch_size=500,
                 experiment=experiment)
tgan.fit(d)

model_path = f'model/berka_{project_name}'

tgan.save(model_path)

num_samples = 100000
new_samples = tgan.sample(num_samples)

p = new_samples.copy()
p[p._get_numeric_data().columns] = p[p._get_numeric_data().columns].astype('int')
p.to_csv(f'samples/berka_sample_{project_name}.csv', index=False)
experiment.log_asset_data(p, file_name=f'sample_{project_name}_{len(p)}', overwrite=False)
experiment.log_dataset_info(name=ds)
