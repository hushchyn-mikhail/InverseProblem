import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from inverse_problem.nn_inversion.transforms import normalize_output
from astropy.io import fits
import numpy as np
import pandas as pd
import pylab


def open_param_file(path, normalize=True):
    refer = fits.open(path)
    # print('Open file with 36 available paramters, 11 will be selected')
    param_list = [1, 2, 3, 6, 8, 7, 33, 10, 5, 12, 13]
    names = [refer[i].header['EXTNAME'] for i in param_list]
    print('\n'.join(names))
    data = np.zeros(shape=(512, 485, 11))
    for i, idx in enumerate(param_list):
        data[:, :, i] = refer[idx].data
    if normalize:
        shape = data.shape
        data = normalize_output(data.reshape(-1, 11)).reshape(shape)

    return data, names


def compute_metrics(refer, predicted, names, save_path=None):
    r2list = []
    mselist = []
    maelist = []
    for i, _ in enumerate(names):
        r2list.append(np.corrcoef(refer[:, :, i].flatten(), predicted[:, :, i].flatten())[0][1] ** 2)
        mselist.append(mean_squared_error(refer[:, :, i].flatten(), predicted[:, :, i].flatten()))
        maelist.append(mean_absolute_error(refer[:, :, i].flatten(), predicted[:, :, i].flatten()))
    df = pd.DataFrame([r2list, mselist, maelist], columns=names, index=['r2', 'mse', 'mae']).T.round(3)
    if save_path:
        df.to_csv(save_path)
    return df


def plot_params(data):
    """Draw all 11 parameters at once
    data: np array (:, :, 11)
    """
    names = ['Field_Strength',
             'Field_Inclination',
             'Field_Azimuth',
             'Doppler_Width',
             'Damping',
             'Line_Strength',
             'Original_Continuum_Intensity',
             'Source_Function_Gradient',
             'Doppler_Shift2',
             'Stray_Light_Fill_Factor',
             'Stray_Light_Shift']

    plt.figure(figsize=(12, 9))
    plt.axis('off')
    for i in range(11):
        plt.subplot(3, 4, i + 1)
        plt.title(names[i])
        plt.imshow(data[:, :, i], cmap='gray')
        plt.axis('off')



def metrics(true, pred):
    print('r2', r2_score(true, pred))
    print('rmse', mean_squared_error(true, pred, squared=False))
    print('mse', mean_squared_error(true, pred))
    print('mae', mean_absolute_error(true, pred))
    print('')


def plot_spectra(pred, true):
    """ Draw prediction and reference"""
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
    sc = axs[0].imshow(pred, cmap='gray')
    fig.colorbar(sc, ax=axs[0])
    axs[0].set_title('Predicted')

    sc = axs[1].imshow(true, cmap='gray')
    fig.colorbar(sc, ax=axs[1])
    axs[1].set_title('True')


def plot_2d(names, refer, predicted_mu, predicted_sigma, index=0):
    '''
        draws 2d graphs:
        1. (x_true - x_pred)/sigma_pred vs x_true,
        2. (x_true - x_pred) vs sigma_pred,
        3. x_true vs sigma_pred.
        param index: number of a graph, from 0 to 2
        return: saves the graph to the ../img/ directory
    '''
    titles_for_saving = ['x_true vs (x_true - x_pred)\sigma_pred',
                       'x_true - x_pred vs sigma_pred',
                       'x_true vs sigma_pred']

    fig, axs = pylab.subplots(3, 4, figsize=(19, 15))
    for i, ax in enumerate(axs.flat[:-1]):
        if index == 0:
            X, Y = refer[:, :, i].flatten(), (refer[:, :, i].flatten() - predicted_mu[:, :, i].flatten()) / predicted_sigma[:, :, i].flatten()
        elif index == 1:
            X, Y = refer[:, :, i].flatten() - predicted_mu[:, :, i].flatten(), predicted_sigma[:, :, i].flatten()
        else:
            X, Y = refer[:, :, i].flatten(), predicted_sigma[:, :, i].flatten()
        ax.set_title(names[i], weight='bold')
        ax.plot(X, Y, 'o', color='red', alpha=0.1, markersize=4, markeredgewidth=0.0)
        ax.axis(ymin=0.8*min(Y), ymax=1.2*max(Y))

    if index == 0:
        fig.supxlabel(r'$x_{true}$')
        fig.supylabel(r'$(x_{true} - x_{pred})/ \sigma_{pred}$')
    elif index == 1:
        fig.supxlabel(r'$x_{true} - x_{pred}$')
        fig.supylabel(r'$\sigma_{pred}$')
    else:
        fig.supxlabel(r'$x_{true}$')
        fig.supylabel(r'$\sigma_{pred}$')

    fig.set_facecolor('xkcd:white')
    fig.delaxes(axs[2][3])
    pylab.tight_layout(pad=3)
    fig.savefig("../img/" + titles_for_saving[index] + ".png")
    pylab.show()


def plot_1d(names, refer, predicted_mu, predicted_sigma, bins=5):
    '''
        draws a histogram: (x_true - x_pred)/sigma_pred
        param bins: number of bins
        return: saves the graph to the ../img/ directory
    '''
    fig, axs = pylab.subplots(3, 4, figsize=(19, 15))
    for i, ax in enumerate(axs.flat[:-1]):
        X = (refer[:, :, i].flatten() - predicted_mu[:, :, i].flatten()) / predicted_sigma[:, :, i].flatten()
        ax.set_title(names[i], weight='bold')
        ax.hist(X, bins=bins)
        ax.axis(xmin=max(0.8 * min(X), -10), xmax=min(1.2 * max(X), 10))

    fig.supxlabel(r'$(x_{true} - x_{pred})/ \sigma_{pred}$')
    fig.set_facecolor('xkcd:white')
    fig.delaxes(axs[2][3])
    pylab.tight_layout(pad=3)
    fig.savefig("../img/" + r'(x_true - x_pred)\sigma_pred' + ".png")
    pylab.show()
