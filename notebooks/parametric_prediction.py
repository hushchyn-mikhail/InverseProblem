import json
from pprint import pprint
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path
from inverse_problem.nn_inversion.main import HyperParams, Model
from inverse_problem import get_project_root
from inverse_problem.milne_edington.me import read_full_spectra
from inverse_problem.nn_inversion.posthoc import compute_metrics, open_param_file, plot_params

ref_id ='1ylXZjKTx0riGwq520f2IuD4k_z3z5piN'
gdd.download_file_from_google_drive(file_id=ref_id, dest_path='../data/reference.fits', showsize=True)

refer_path = '../data/reference.fits'
refer, names = open_param_file(refer_path)

path_to_json = '../res_experiments/hps_partly_independent_mlp.json'
with open(path_to_json) as f:
    json_data = json.load(f)

params = fits.open('../data/parameters_base.fits')[0].data

hps = HyperParams.from_file(path_to_json=path_to_json)
# hps.activation ='sigmoid'
hps.batch_size = 1024
hps.n_epochs = 10
hps.lr = 0.0001

model = Model(hps)

history = model.train(
    data_arr=params[:15000],
    logdir = "../",
    path_to_save='../partly_ind_mlp.pt')

predicted_partly = model.predict_refer('../data/reference.fits')
df_partly = compute_metrics(refer, predicted_partly, names, save_path = '../partly_pred.csv')
print(df_partly)
plot_params(predicted_partly)
