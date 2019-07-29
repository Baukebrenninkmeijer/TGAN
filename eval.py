import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import copy
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import scipy.stats as stats
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from dython.nominal import *
import warnings
from sklearn.exceptions import ConvergenceWarning

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def plot_correlation_difference(real: pd.DataFrame, fake: pd.DataFrame, plot_diff=True, cat_cols=None, *args, **kwargs):
    if cat_cols is None:
        cat_cols = real.select_dtypes(['object', 'category'])
    if plot_diff:
        fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    real_corr = associations(real, nominal_columns=cat_cols, return_results=True, plot=True, theil_u=True,
                             mark_columns=True, ax=ax[0], **kwargs)
    fake_corr = associations(fake, nominal_columns=cat_cols, return_results=True, plot=True, theil_u=True,
                             mark_columns=True, ax=ax[1], **kwargs)

    if plot_diff:
        diff = abs(real_corr - fake_corr)
        sns.set(style="white")
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(diff, ax=ax[2], cmap=cmap, vmax=.3, square=True, annot=kwargs.get('annot', True), center=0,
                    linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')

    titles = ['Real', 'Fake', 'Difference'] if plot_diff else ['Real', 'Fake']
    for i, label in enumerate(titles):
        title_font = {'size': '18'}
        ax[i].set_title(label, **title_font)
    plt.tight_layout()
    plt.show()


def matrix_distance_abs(ma, mb):
    return np.sum(np.abs(np.subtract(ma, mb)))


def matrix_distance_euclidian(ma, mb):
    return np.sqrt(np.sum(np.power(np.subtract(ma, mb), 2)))


def cdf(data_r, data_f, xlabel, ylabel, ax=None):
    """
    Plot continous density function on optionally given ax. If no ax, cdf is plotted and shown.
    :param data_r: Series with real data
    :param data_f: Series with fake data
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param ax: axis to plot on
    """
    x1 = np.sort(data_r)
    x2 = np.sort(data_f)
    y = np.arange(1, len(data_r) + 1) / len(data_r)

    ax = ax if ax else plt.subplots()[1]

    axis_font = {'size': '14'}
    ax.set_xlabel(xlabel, **axis_font)
    ax.set_ylabel(ylabel, **axis_font)

    ax.grid()
    ax.plot(x1, y, marker='o', linestyle='none', label='Real', ms=8)
    ax.plot(x2, y, marker='o', linestyle='none', label='Fake', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    # ax.set_xticks(ind)
    if data_r.dtypes == 'object':
        ax.set_xticklabels(data_r.value_counts().sort_index().index, rotation='vertical')

    if ax is None:
        plt.show()


def categorical_distribution(real, fake, xlabel, ylabel, col=None, ax=None):
    ax = ax if ax else plt.subplots()[1]
    if col is not None:
        real = real[col]
        fake = fake[col]
    y_r = real.value_counts().sort_index() / len(real)
    y_f = fake.value_counts().sort_index() / len(fake)

    # width = 0.35  # the width of the bars
    ind = np.arange(len(y_r.index))

    ax.grid()
    yr_cumsum = y_r.cumsum()
    yf_cumsum = y_f.cumsum()
    values = yr_cumsum.values.tolist() + yf_cumsum.values.tolist()
    real = [1 for _ in range(len(yr_cumsum))] + [0 for _ in range(len(yf_cumsum))]
    classes = yr_cumsum.index.tolist() + yf_cumsum.index.tolist()
    data = pd.DataFrame({'values': values,
                         'real': real,
                         'class': classes})
    paper_rc = {'lines.linewidth': 8}
    sns.set_context("paper", rc=paper_rc)
    #     ax.plot(x=yr_cumsum.index.tolist(), y=yr_cumsum.values.tolist(), ms=8)
    sns.lineplot(y='values', x='class', data=data, ax=ax, hue='real')
    #     ax.bar(ind - width / 2, y_r.values, width, label='Real')
    #     ax.bar(ind + width / 2, y_f.values, width, label='Fake')

    ax.set_ylabel('Distributions per variable')

    axis_font = {'size': '18'}
    ax.set_xlabel(xlabel, **axis_font)
    ax.set_ylabel(ylabel, **axis_font)

    ax.set_xticks(ind)
    ax.set_xticklabels(y_r.index, rotation='vertical')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)


def mean_absolute_error(ma, mb):
    return np.mean(np.abs(np.subtract(ma, mb)))


def euclidean_distance(ma, mb):
    return np.sqrt(np.sum(np.power(np.subtract(ma, mb), 2)))


def column_correlations(dataset_a, dataset_b, categorical_columns, theil_u=True):
    if categorical_columns is None:
        categorical_columns = list()
    elif categorical_columns == 'all':
        categorical_columns = dataset_a.columns
    assert dataset_a.columns.tolist() == dataset_b.columns.tolist()
    corr = pd.DataFrame(columns=dataset_a.columns, index=['correlation'])

    for column in dataset_a.columns.tolist():
        if column in categorical_columns:
            if theil_u:
                corr[column] = theils_u(dataset_a[column].sort_values(), dataset_b[column].sort_values())
            else:
                corr[column] = cramers_v(dataset_a[column].sort_values(), dataset_b[column].sort_vaues())
        else:
            corr[column], _ = ss.pearsonr(dataset_a[column].sort_values(), dataset_b[column].sort_values())
    corr.fillna(value=np.nan, inplace=True)
    correlation = np.mean(corr.values.flatten())
    return correlation


def associations(dataset, nominal_columns=None, mark_columns=False, theil_u=False, plot=True,
                 return_results=False, **kwargs):
    """
    Adapted from: https://github.com/shakedzy/dython

    Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
    continuous features using:
     - Pearson's R for continuous-continuous cases
     - Correlation Ratio for categorical-continuous cases
     - Cramer's V or Theil's U for categorical-categorical cases
    :param dataset: NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    :param nominal_columns: string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
        columns are categorical, or None (default) to state none are categorical
    :param mark_columns: Boolean (default: False)
        if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
        continuous), as provided by nominal_columns
    :param theil_u: Boolean (default: False)
        In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
    :param plot: Boolean (default: True)
        If True, plot a heat-map of the correlation matrix
    :param return_results: Boolean (default: False)
        If True, the function will return a Pandas DataFrame of the computed associations
    :param kwargs:
        Arguments to be passed to used function and methods
    :return: Pandas DataFrame
        A DataFrame of the correlation/strength-of-association between all features
    """

    dataset = convert(dataset, 'dataframe')
    columns = dataset.columns
    if nominal_columns is None:
        nominal_columns = list()
    elif nominal_columns == 'all':
        nominal_columns = columns
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0, len(columns)):
        for j in range(i, len(columns)):
            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                if columns[i] in nominal_columns:
                    if columns[j] in nominal_columns:
                        if theil_u:
                            corr[columns[j]][columns[i]] = theils_u(dataset[columns[i]], dataset[columns[j]])
                            corr[columns[i]][columns[j]] = theils_u(dataset[columns[j]], dataset[columns[i]])
                        else:
                            cell = cramers_v(dataset[columns[i]], dataset[columns[j]])
                            corr[columns[i]][columns[j]] = cell
                            corr[columns[j]][columns[i]] = cell
                    else:
                        cell = correlation_ratio(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                else:
                    if columns[j] in nominal_columns:
                        cell = correlation_ratio(dataset[columns[j]], dataset[columns[i]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                    else:
                        cell, _ = ss.pearsonr(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in
                          columns]
        corr.columns = marked_columns
        corr.index = marked_columns
    if plot:
        if kwargs.get('ax') is None:
            plt.figure(figsize=kwargs.get('figsize', None))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.set(style="white")
        sns.heatmap(corr, annot=kwargs.get('annot', True), fmt=kwargs.get('fmt', '.2f'), cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=kwargs.get('ax', None))
        if kwargs.get('ax') is None:
            plt.show()
    if return_results:
        return corr


def numerical_encoding(dataset, nominal_columns='all', drop_single_label=False, drop_fact_dict=True):
    """
    Adapted from: https://github.com/shakedzy/dython

    Encoding a data-set with mixed data (numerical and categorical) to a numerical-only data-set,
    using the following logic:
    * categorical with only a single value will be marked as zero (or dropped, if requested)
    * categorical with two values will be replaced with the result of Pandas `factorize`
    * categorical with more than two values will be replaced with the result of Pandas `get_dummies`
    * numerical columns will not be modified
    **Returns:** DataFrame or (DataFrame, dict). If `drop_fact_dict` is True, returns the encoded DataFrame.
    else, returns a tuple of the encoded DataFrame and dictionary, where each key is a two-value column, and the
    value is the original labels, as supplied by Pandas `factorize`. Will be empty if no two-value columns are
    present in the data-set
    Parameters
    ----------
    dataset : NumPy ndarray / Pandas DataFrame
        The data-set to encode
    nominal_columns : sequence / string
        A sequence of the nominal (categorical) columns in the dataset. If string, must be 'all' to state that
        all columns are nominal. If None, nothing happens. Default: 'all'
    drop_single_label : Boolean, default = False
        If True, nominal columns with a only a single value will be dropped.
    drop_fact_dict : Boolean, default = True
        If True, the return value will be the encoded DataFrame alone. If False, it will be a tuple of
        the DataFrame and the dictionary of the binary factorization (originating from pd.factorize)
    """
    dataset = convert(dataset, 'dataframe')
    if nominal_columns is None:
        return dataset
    elif nominal_columns == 'all':
        nominal_columns = dataset.columns
    converted_dataset = pd.DataFrame()
    binary_columns_dict = dict()
    for col in dataset.columns:
        if col not in nominal_columns:
            converted_dataset.loc[:, col] = dataset[col]
        else:
            unique_values = pd.unique(dataset[col])
            if len(unique_values) == 1 and not drop_single_label:
                converted_dataset.loc[:, col] = 0
            elif len(unique_values) == 2:
                converted_dataset.loc[:, col], binary_columns_dict[col] = pd.factorize(dataset[col])
            else:
                dummies = pd.get_dummies(dataset[col], prefix=col)
                converted_dataset = pd.concat([converted_dataset, dummies], axis=1)
    if drop_fact_dict:
        return converted_dataset
    else:
        return converted_dataset, binary_columns_dict


def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)


class DataEvaluator:
    def __init__(self, real, fake, unique_thresh=20, metric='kendalltau', verbose=False, n_samples=None):
        if isinstance(real, np.ndarray):
            real = pd.DataFrame(real)
            fake = pd.DataFrame(fake)
        assert isinstance(real, pd.DataFrame), f'Make sure you either pass a Pandas DataFrame or Numpy Array'

        self.unique_thresh = unique_thresh
        self.numerical_columns = [column for column in real._get_numeric_data().columns if
                                  len(real[column].unique()) > unique_thresh]
        self.categorical_columns = [column for column in real.columns if column not in self.numerical_columns]
        self.real = real
        self.fake = fake
        self.comparison_metric = getattr(stats, metric)
        self.verbose = verbose

        if n_samples is None:
            self.n_samples = min(len(self.real), len(self.fake))
        else:
            self.n_samples = n_samples
        self.real = self.real.sample(self.n_samples)
        self.fake = self.fake.sample(self.n_samples)
        assert len(self.real) == len(self.fake), f'len(real) != len(fake)'

    def plot_mean_std(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Absolute Log Mean and STDs of numeric data\n', fontsize=16)

        real = self.real._get_numeric_data()
        fake = self.fake._get_numeric_data()
        real_mean = np.log10(np.add(abs(real.mean()).values, 1e-5))
        fake_mean = np.log10(np.add(abs(fake.mean()).values, 1e-5))
        sns.scatterplot(x=real_mean,
                        y=fake_mean,
                        ax=ax[0])
        line = np.arange(min(real_mean + [-5]), max(real_mean + [10]))
        sns.lineplot(x=line, y=line, ax=ax[0])
        ax[0].set_title('Means of real and fake data')
        ax[0].set_xlabel('real data mean (log)')
        ax[0].set_ylabel('fake data mean (log)')

        real_std = np.log10(np.add(real.std().values, 1e-5))
        fake_std = np.log10(np.add(fake.std().values, 1e-5))
        line = np.arange(min(real_std + [-5]), max(real_std + [10]))
        sns.scatterplot(x=real_std,
                        y=fake_std,
                        ax=ax[1])
        sns.lineplot(x=line, y=line, ax=ax[1])
        ax[1].set_title('Stds of real and fake data')
        ax[1].set_xlabel('real data std (log)')
        ax[1].set_ylabel('fake data std (log)')
        plt.show()

    def plot_cumsums(self):
        nr_charts = len(self.real.columns)
        nr_cols = 4
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows


        # Increase the length of plots if the labels are long
        lengths = []
        for d in self.real.select_dtypes(include=['object']):
            lengths.append(max([len(x.strip()) for x in self.real[d].unique().tolist()]))
        max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Cumulative Sums per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.real.columns):
            r = self.real[col]
            f = self.fake.iloc[:, self.real.columns.tolist().index(col)]
            cdf(r, f, col, 'Cumsum', ax=axes[i])
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.show()

    def plot_correlation_difference(self, plot_diff=True, *args, **kwargs):
        plot_correlation_difference(self.real, self.fake, cat_cols=self.categorical_columns, plot_diff=plot_diff, *args,
                                    **kwargs)

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

        real_corr = associations(self.real, nominal_columns=self.categorical_columns, return_results=True, theil_u=True, plot=False)
        fake_corr = associations(self.fake, nominal_columns=self.categorical_columns, return_results=True, theil_u=True, plot=False)

        return distance_func(
            real_corr.values,
            fake_corr.values
        )

    def plot_2d(self):
        """
        Plot the first two components of a PCA of the numeric columns of real and fake.
        """
        real = numerical_encoding(self.real, nominal_columns=self.categorical_columns)
        fake = numerical_encoding(self.fake, nominal_columns=self.categorical_columns)
        pca_r = PCA(n_components=2)
        pca_f = PCA(n_components=2)

        real_t = pca_r.fit_transform(real)
        fake_t = pca_f.fit_transform(fake)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('First two components of PCA', fontsize=16)
        sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
        sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
        ax[0].set_title('Real data')
        ax[1].set_title('Fake data')
        plt.show()

    def get_duplicates(self):
        """
        Check whether any real values occur in the fake data
        :return: Dataframe containing the duplicates
        """
        df = pd.concat([self.real, self.fake])
        duplicates = df[df.duplicated(keep=False)]
        return duplicates

    def pca_correlation(self):
        pca_r = PCA(n_components=5)
        pca_f = PCA(n_components=5)

        real = numerical_encoding(self.real, nominal_columns=self.categorical_columns)
        fake = numerical_encoding(self.fake, nominal_columns=self.categorical_columns)

        pca_r.fit(real)
        pca_f.fit(fake)
        if self.verbose:
            results = pd.DataFrame({'real': pca_r.explained_variance_, 'fake': pca_f.explained_variance_})
            print(f'\nTop 5 PCA components:')
            print(results.to_string())
        slope, int, corr, p, _ = scipy.stats.linregress(pca_r.explained_variance_, pca_f.explained_variance_)
        return corr

    def fit_estimators(self):
        """
        Fit self.r_estimators and self.f_estimators to real and fake data, respectively.
        """
        if self.verbose:
            print(f'\nFitting real')
        for i, c in enumerate(self.r_estimators):
            if self.verbose:
                print(f'{i + 1}: {type(c).__name__}')
            c.fit(self.real_x_train, self.real_y_train)

        if self.verbose:
            print(f'\nFitting fake')
        for i, c in enumerate(self.f_estimators):
            if self.verbose:
                print(f'{i + 1}: {type(c).__name__}')
            c.fit(self.fake_x_train, self.fake_y_train)

    def score_estimators(self):
        """
        Get F1 scores of self.r_estimators and self.f_estimators on the fake and real data, respectively.
        :return:
        """
        from sklearn.metrics import mean_squared_error

        if self.target_type == 'class':
            r2r = [f1_score(self.real_y_test, clf.predict(self.real_x_test), average='micro') for clf in self.r_estimators]
            f2f = [f1_score(self.fake_y_test, clf.predict(self.fake_x_test), average='micro') for clf in self.f_estimators]

            # Calculate test set accuracies on the other dataset
            r2f = [f1_score(self.fake_y_test, clf.predict(self.fake_x_test), average='micro') for clf in self.r_estimators]
            f2r = [f1_score(self.real_y_test, clf.predict(self.real_x_test), average='micro') for clf in self.f_estimators]
            index = [f'real_data_{classifier}_F1' for classifier in self.estimator_names] + \
                    [f'fake_data_{classifier}_F1' for classifier in self.estimator_names]
            results = pd.DataFrame({'real': r2r + r2f, 'fake': f2r + f2f}, index=index)

        elif self.target_type == 'regr':
            r2r = [mean_squared_error(self.real_y_test, clf.predict(self.real_x_test)) for clf in self.r_estimators]
            f2f = [mean_squared_error(self.fake_y_test, clf.predict(self.fake_x_test)) for clf in self.f_estimators]

            # Calculate test set accuracies on the other dataset
            r2f = [mean_squared_error(self.fake_y_test, clf.predict(self.fake_x_test)) for clf in self.r_estimators]
            f2r = [mean_squared_error(self.real_y_test, clf.predict(self.real_x_test)) for clf in self.f_estimators]
            index = [f'real_data_{classifier}_F1' for classifier in self.estimator_names] + \
                    [f'fake_data_{classifier}_F1' for classifier in self.estimator_names]
            results = pd.DataFrame({'real': r2r + r2f, 'fake': f2r + f2f}, index=index)
        else:
            raise Exception(f'self.target_type should be either \'class\' or \'regr\', but is {self.target_type}.')
        return results

    def visual_evaluation(self, plot=True, **kwargs):
        if plot:
            self.plot_mean_std()
            self.plot_cumsums()
            self.plot_correlation_difference(**kwargs)
            self.plot_2d()

        # values = []
        # metrics = [
        #     'Nr. of duplicate rows',
        #     'Euclidean distance correlations',
        #     'MAE distance correlations',
        # ]
        # values.append(len(self.get_duplicates()))
        # values.append(self.correlation_distance(how='euclidean'))
        # values.append(self.correlation_distance(how='mae'))
        #
        # summary = pd.DataFrame({'values': values})
        # summary.index = metrics
        # return summary

    def statistical_evaluation(self):
        total_metrics = pd.DataFrame()
        for ds_name in ['real', 'fake']:
            ds = getattr(self, ds_name)
            metrics = {}
            num_ds = ds[self.numerical_columns]

            # Basic statistical properties
            for idx, value in num_ds.mean().items():
                metrics[f'mean_{idx}'] = value
            for idx, value in num_ds.median().items():
                metrics[f'median_{idx}'] = value
            for idx, value in num_ds.std().items():
                metrics[f'std_{idx}'] = value
            for idx, value in num_ds.var().items():
                metrics[f'variance_{idx}'] = value
            total_metrics[ds_name] = metrics.values()

        total_metrics.index = metrics.keys()
        self.statistical_results = total_metrics
        if self.verbose:
            print('\nBasic statistical attributes:')
            print(total_metrics.to_string())
        corr, p = self.comparison_metric(total_metrics['real'], total_metrics['fake'])
        return corr

    def correlation_correlation(self):
        total_metrics = pd.DataFrame()
        for ds_name in ['real', 'fake']:
            ds = getattr(self, ds_name)
            corr_df = associations(ds, nominal_columns=self.categorical_columns, return_results=True, theil_u=True, plot=False)
            values = corr_df.values
            values = values[~np.eye(values.shape[0], dtype=bool)].reshape(values.shape[0],-1)
            total_metrics[ds_name] = values.flatten()

        self.correlation_correlations = total_metrics
        corr, p = self.comparison_metric(total_metrics['real'], total_metrics['fake'])
        if self.verbose:
            print('\nColumn correlation between datasets:')
            print(total_metrics.to_string())
        return corr

    def estimator_evaluation(self, target_col, target_type='class', n_samples=None):
        self.target_col = target_col
        self.target_type = target_type

        # Convert both datasets to numerical representations and split x and  y
        real_x = numerical_encoding(self.real.drop([target_col], axis=1), nominal_columns=self.categorical_columns)
        columns = sorted(real_x.columns.tolist())
        real_x = real_x[columns]
        fake_x = numerical_encoding(self.fake.drop([target_col], axis=1), nominal_columns=self.categorical_columns)
        for col in columns:
            if col not in fake_x.columns.tolist():
                fake_x[col] = 0
        fake_x = fake_x[columns]
        assert real_x.columns.tolist() == fake_x.columns.tolist(), f'real and fake columns are different: \n{real_x.columns}\n{fake_x.columns}'

        if self.target_type == 'class':
            # Encode real and fake target the same
            real_y, uniques = pd.factorize(self.real[target_col])
            mapping = {key: value for value, key in enumerate(uniques)}
            fake_y = [mapping.get(key) for key in self.fake[target_col].tolist()]
        else:
            real_y = self.real[target_col]
            fake_y = self.fake[target_col]

        # split real and fake into train and test sets
        self.real_x_train, self.real_x_test, self.real_y_train, self.real_y_test = train_test_split(real_x, real_y, test_size=0.2)
        self.fake_x_train, self.fake_x_test, self.fake_y_train, self.fake_y_test = train_test_split(fake_x, fake_y, test_size=0.2)

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import ElasticNet

        if target_type == 'regr':
            self.estimators = [
                RandomForestRegressor(n_estimators=20, max_depth=5),
                Lasso(),
                Ridge(alpha=1.0),
                ElasticNet(),
            ]
        elif target_type == 'class':
            self.estimators = [
                SGDClassifier(max_iter=100, tol=1e-3),
                LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=500),
                RandomForestClassifier(),
                DecisionTreeClassifier(),
                MLPClassifier([50, 50], solver='adam', activation='relu', learning_rate='adaptive'),
            ]
        else:
            raise Exception(f'target_type must be \'regr\' or \'class\'')


        self.r_estimators = copy.deepcopy(self.estimators)
        self.f_estimators = copy.deepcopy(self.estimators)
        self.estimator_names = [type(clf).__name__ for clf in self.estimators]

        for estimator in self.estimators:
            assert hasattr(estimator, 'fit')
            assert hasattr(estimator, 'score')

        self.fit_estimators()
        self.estimators_scores = self.score_estimators()
        # if self.verbose:
        print('\nClassifier F1-scores:') if self.target_type == 'class' else print('\nRegressor MSE-scores:')
        print(self.estimators_scores.to_string())
        corr, p = self.comparison_metric(self.estimators_scores['real'], self.estimators_scores['fake'])
        return corr

    def evaluate(self, target_col, target_type='class', metric=None, verbose=None):
        """
        Determine correlation between attributes from the real and fake dataset using a given metric.
        All metrics from scipy.stats are available.
        :param target_col: column to use for predictions with estimators
        :param n_samples: the number of samples to use for the estimators. Training time scales mostly linear
        :param metric: scoring metric for the attributes. By default Kendall Tau ranking is used. Alternatives
            include Spearman rho (scipy.stats.spearmanr) ranking.
        """
        if verbose is not None:
            self.verbose = verbose
        if metric is not None:
            self.comparison_metric = metric

        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        pd.options.display.float_format = '{:,.4f}'.format

        print(f'\nCorrelation metric: {self.comparison_metric.__name__}')

        basic_statistical = self.statistical_evaluation()  # 2 columns -> Kendall Tau -> correlation coefficient
        correlation_correlation = self.correlation_correlation()  # 2 columns -> Kendall Tau -> Correlation coefficient
        column_correlation = column_correlations(self.real, self.fake, self.categorical_columns)  # 1 column -> Mean
        estimators = self.estimator_evaluation(target_col=target_col, n_samples=self.n_samples, target_type=target_type)  # 1 2 columns -> Kendall Tau -> Correlation coefficient
        pca_variance = self.pca_correlation()  # 1 number

        all_results = {
            'basic statistics': basic_statistical,
            'Correlation column correlations': correlation_correlation,
            'Mean Correlation between fake and real columns': column_correlation,
            'Mean correlation classifier F1': estimators,
            'Correlation 5 PCA components': pca_variance,
        }
        total_result = np.mean(list(all_results.values()))
        all_results['Duplicate data between sets'] = len(self.get_duplicates())
        all_results['Total Result'] = total_result
        all_results_df = pd.DataFrame({'Result': list(all_results.values())}, index=list(all_results.keys()))

        print(f'\nResults:\nNumber of duplicate rows is ignored for total score.')
        print(all_results_df.to_string())
