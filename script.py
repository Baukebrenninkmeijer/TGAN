import pandas as pd
d = pd.read_csv('../data/berka/berka_cat.csv', sep=';')
d = d.drop(['trans_bank_partner', 'trans_account_partner'], axis=1)
continuous_columns = [0, 1, 2, 3, 9]
from tgan.model import TGANModel

tgan = TGANModel(continuous_columns, restore_session=False,  max_epoch=50, steps_per_epoch=1000, batch_size=1000)
tgan.fit(d)

model_path = 'demo/my_model'

tgan.save(model_path)