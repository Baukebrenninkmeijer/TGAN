import pandas as pd
d = pd.read_csv('../data/berka/berka_cat.csv', sep=';')
d = d.drop(['trans_bank_partner', 'trans_account_partner'], axis=1)
continuous_columns = [0, 1, 2, 3, 7]
from tgan.model import TGANModel

tgan = TGANModel(continuous_columns, 
                 restore_session=False,  
                 max_epoch=100, 
                 steps_per_epoch=1000, 
                 batch_size=1000,
                 comet_ml_key='49HGMPyIKjokHwg2pVOKWTG67')
tgan.fit(d)

model_path = 'model/berka_wgan_100x1000'

tgan.save(model_path)

num_samples = 10000
new_samples = tgan.sample(num_samples)

p = new_samples.copy()
p[p._get_numeric_data().columns] = p[p._get_numeric_data().columns].astype('int')
p.to_csv('samples/berka_sample_wgan.csv', index=False)
