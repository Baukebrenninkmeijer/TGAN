import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def plot_var_cor(x, ax=None, ret=False, *args, **kwargs):
    if isinstance(x, pd.DataFrame):
        corr = x.corr().values
    elif isinstance(x, np.ndarray):
        corr = np.corrcoef(x, rowvar=False)
    else:
        raise Exception('Unknown datatype given. Make sure a Pandas DataFrame or Numpy Array is passed.')
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    if type(ax) is None:
        f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, *args, **kwargs)
    if ret:
        return corr


def plot_corr_diff(real, fake, plot_diff=False, *args, **kwargs):
    if plot_diff:
        fig, ax = plt.subplots(1, 3, figsize=(22, 8))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(22, 8))

    real_corr = plot_var_cor(real, ax=ax[0], ret=True)
    fake_corr = plot_var_cor(fake, ax=ax[1], ret=True)

    if plot_diff:
        diff = abs(real_corr - fake_corr)
        sns.set(style="white")

        # Generate a mask for the upper triangle
        mask = np.zeros_like(diff, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(diff, ax=ax[2], mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, *args, **kwargs)

    titles = ['y', 'y_hat', 'diff'] if plot_diff else ['y', 'y_hat']
    for i, label in enumerate(titles):
        title_font = {'size': '18'}
        axis_font = {'size': '14'}

        ax[i].set_title(label, **title_font)
        ax[i].set_yticklabels(real.columns.values, rotation='horizontal', **axis_font)
        ax[i].set_xticks(list(np.arange(0.5, real_corr.shape[0] + 0.5, 1)))
        ax[i].set_xticklabels(real.columns.values, rotation='vertical', **axis_font)
        plt.tight_layout()
    plt.show()


def matrix_distance_abs(ma, mb):
    return np.sum(np.abs(np.subtract(ma, mb)))


def matrix_distance_euclidian(ma, mb):
    return np.sqrt(np.sum(np.power(np.subtract(ma, mb), 2)))


def cdf(data_r, data_f, xlabel, ylabel, ax=None):
    x1 = np.sort(data_r)
    x2 = np.sort(data_f)
    y = np.arange(1, len(data_r) + 1) / len(data_r)

    if ax is None:
        fig, fig_ax = plt.subplots()
    else:
        fig_ax = ax

    axis_font = {'size': '18'}
    fig_ax.set_xlabel(xlabel, **axis_font)
    fig_ax.set_ylabel(ylabel, **axis_font)

    fig_ax.grid()
    fig_ax.margins(0.02)

    fig_ax.plot(x1, y, marker='o', linestyle='none', label='Real Data', ms=8)
    fig_ax.plot(x2, y, marker='o', linestyle='none', label='Fake Data', alpha=0.5, ms=5)

    fig_ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    if ax is None:
        plt.show()


def mean_absolute_error(ma, mb):
    return np.mean(np.abs(np.subtract(ma, mb)))


def euclidean_distance(ma, mb):
    return np.sqrt(np.sum(np.power(np.subtract(ma, mb), 2)))


class BaseDataEvaluator:
    def __init__(self, real, fake, unique_thresh=20):
        if isinstance(real, np.ndarray):
            real = pd.DataFrame(real)
            fake = pd.DataFrame(fake)
        assert isinstance(real, pd.DataFrame), f'Make sure you either pass a Pandas DataFrame or Numpy Array'
        self.numerical_columns = [column for column in self.real._get_numeric_data().columns if self.real[column].unique > unique_thresh]
        self.categorical_columns = [column for column in self.real.columns if column not in self.numerical_columns]
        self.real = real
        self.fake = fake
        self.real_dummy = None
        self.fake_dummy = None

    def to_dummies(self):
        real_dummy = self.real
        dummies = pd.get_dummies(real_dummy, self.categorical_columns)
        real_dummy = pd.concat([real_dummy, dummies], axis=1, sort=False)
        real_dummy = real_dummy.drop([self.categorical_columns], axis=1)
        assert len(real_dummy) == len(self.real), f'Length of real and dummy data differ'
        self.real_dummy = real_dummy

        fake_dummy = self.fake
        dummies = pd.get_dummies(fake_dummy, self.categorical_columns)
        fake_dummy = pd.concat([fake_dummy, dummies], axis=1, sort=False)
        fake_dummy = fake_dummy.drop([self.categorical_columns], axis=1)
        assert len(fake_dummy) == len(self.fake), f'Length of real and dummy data differ'
        self.fake_dummy = fake_dummy

    def plot_stats(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        real = self.real._get_numeric_data()
        fake = self.fake._get_numeric_data()
        real_num_mean = np.log10(np.add(real.mean().values, 1e-5))
        fake_num_mean = np.log10(np.add(fake.mean().values, 1e-5))
        sns.scatterplot(x=real_num_mean,
                        y=fake_num_mean,
                        ax=ax[0])
        print(len(real_num_mean))
        line = np.arange(min(real_num_mean + [-5]), max(real_num_mean + [10]))
        sns.lineplot(x=line, y=line, ax=ax[0])
        ax[0].set_title('Means of real and fake data')
        ax[0].set_xlabel('real data mean (log)')
        ax[0].set_ylabel('fake data mean (log)')

        real_cat_std = np.log10(np.add(real.std().values, 1e-5))
        fake_cat_std = np.log10(np.add(fake.std().values, 1e-5))
        line = np.arange(min(real_cat_std + [-5]), max(real_cat_std + [10]))
        sns.scatterplot(x=real_cat_std,
                        y=fake_cat_std,
                        ax=ax[1])
        sns.lineplot(x=line, y=line, ax=ax[1])
        ax[1].set_title('Stds of real and fake data')
        ax[1].set_xlabel('real data std (log)')
        ax[1].set_ylabel('fake data std (log)')
        plt.show()

    def plot_cumsums(self):
        nr_charts = len(self.fake._get_numeric_data().columns)
        nr_cols = 4
        nr_rows = nr_charts // nr_cols + 1
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, 4 * nr_rows))
        axes = ax.flatten()
        for i, col in enumerate(self.fake._get_numeric_data().columns):
            r = self.real[col]
            f = self.fake.iloc[:, self.real.columns.tolist().index(col)]
            cdf(r, f, col, 'Cumsum', ax=axes[i])
        plt.tight_layout()
        plt.show()

    def plot_correlation_difference(self, plot_diff=True, *args, **kwargs):
        plot_corr_diff(self.real, self.fake, plot_diff, *args, **kwargs)

    def correlation_distance(self, how='euclidean'):
        """
        Calculate distance between correlation matrices with certain metric.
        Metric options are: euclidean, mae (mean absolute error)
        :param how: metric to measure distance
        :return: distance
        """
        distance_func = None
        if how == 'euclidean':
            distance_func = euclidean_distance
        elif how == 'mae':
            distance_func = mean_absolute_error

        assert distance_func is not None, f'Distance measure was None. Please select a measure from [euclidean, mae]'

        return distance_func(
            self.real.corr().fillna(0).values,
            self.fake.corr().fillna(0).values
        )

    def plot_2d(self):
        """
        Plot the first two components of a PCA of the numeric columns of real and fake.
        """
        numeric_columns = self.real._get_numeric_data().columns
        real = self.fake[numeric_columns]
        fake = self.fake[numeric_columns]
        pca_r = PCA(n_components=2)
        pca_f = PCA(n_components=2)

        real_t = pca_r.fit_transform(real)
        fake_t = pca_f.fit_transform(fake)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
        sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
        ax[0].set_title('Real data')
        ax[1].set_title('Fake data')
        plt.show()

    def get_duplicates(self):
        df = pd.concat([self.real, self.fake])
        duplicates = df[df.duplicated(keep=False)]
        return duplicates

    def summary(self, plot=True, as_dict=False):
        if plot:
            self.plot_stats()
            self.plot_cumsums()
            self.plot_correlation_difference()

        rows = []
        data = {}

        metric = '# duplicate rows'
        value = len(self.get_duplicates())
        data[metric] = value
        rows.append({'metric': metric, 'value': value})

        metric = 'Euclidean distance correlations'
        value = self.correlation_distance(how='euclidean')
        data[metric] = value
        rows.append({'metric': metric, 'value': value})

        metric = 'MAE distance correlations'
        value = self.correlation_distance(how='mae')
        data[metric] = value
        rows.append({'metric': metric, 'value': value})

        if as_dict:
            return data

        summary = pd.DataFrame(rows).set_index('metric')
        return summary


class ModelDataEvaluator:
    def __init__(self, real, fake, target_col):
        self.real = real
        self.fake = fake
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        self.classifiers = []
        self.target_col = target_col
