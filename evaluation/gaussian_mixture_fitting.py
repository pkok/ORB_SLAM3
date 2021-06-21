import math
import os.path
import random
import sys

import numpy as np
import pandas as pd
import data_tools

from sklearn.mixture import GaussianMixture

RANDOM_SEED=1

def best_gaussian_mixture(train_data: pd.DataFrame,
                          validation_data: pd.DataFrame,
                          max_components=10,
                          min_components=1,
                          criterion='bic'): 
    """
    Returns the Gaussian mixture with the best number of components.

    It fits a Gaussian mixture model for a number of components from
    min_components up to and including max_components, and selects the one
    with the lowest Akaike information criterion (AIC) or Bayesian information
    criterion (BIC).

    It returns that model, and a DataFrame with weights, means, covariances,
    the chosen criterion and relative likelihood w.r.t. the minimum criterion
    score.
    """
    components = range(min_components, max_components+1)
    columns = ['criterion', 'logRL']
    for c in range(1, max_components+1):
        columns.append(f'w_{c}')
        columns.append(f'μ_{c}')
        columns.append(f'σ_{c}')
    df = pd.DataFrame(index=components,
                      columns=columns)

    t = np.array(train_data).reshape(-1, 1)
    v = np.array(validation_data).reshape(-1, 1)

    crit_eval = lambda model: model.bic(v)
    if criterion == 'aic':
        crit_eval = lambda model: model.aic(v)
    elif criterion != 'bic':
        raise ValueError(f"criterion should be either 'aic' or 'bic', not {criterion}")

    best_model = None
    criterion_min = float('inf')

    for c in components:
        try:
            mixture_model = GaussianMixture(n_components=c,
                                            random_state=RANDOM_SEED)
            mixture_model.fit(t)
            if mixture_model.converged_:
                criterion = crit_eval(mixture_model)
                if criterion < criterion_min:
                    best_model = mixture_model
                    criterion_min = criterion
                df.at[c, 'criterion'] = criterion
                weights = mixture_model.weights_
                means = mixture_model.means_
                covariances = mixture_model.covariances_[:, 0, 0]
                for i, (w, m, s) in enumerate(zip(mixture_model.weights_, 
                                                  mixture_model.means_,
                                                  mixture_model.covariances_)):
                    df.at[c, f'w_{i+1}'] = w
                    df.at[c, f'μ_{i+1}'] = m[0]
                    df.at[c, f'σ_{i+1}'] = s[0][0]

        except:
            pass

    for c in components:
        criterion_c = df.at[c, 'criterion']
        df.at[c, 'logRL'] = (criterion_min - criterion_c)/2

    return best_model, df


def examine_dataset(dataset,
                    mode, 
                    min_components=1,
                    max_components=10,
                    criterion='bic'):
    print(f"{dataset} - {mode} - {criterion.upper()}")

    filename = os.path.join(data_tools.DATA_DIR,
                            data_tools.DATA_FILE_PATTERN.format(dataset=dataset,
                                                                mode=mode))
    data = data_tools.normalize_data(pd.read_csv(filename))

    for column in data.columns:
        valid_data = data[column].loc[~data[column].isnull()]
        # Shuffle data for no dependence between samples in either
        # training or validation set.
        valid_data = valid_data.sample(frac=1,
                                       random_state=RANDOM_SEED)

        n_obs = len(valid_data)
        validation_size = int(0.2 * n_obs)
        validation_data = valid_data[:validation_size]
        train_data = valid_data[validation_size:]

        mixture_model, df = best_gaussian_mixture(train_data, 
                                                  validation_data,
                                                  min_components=min_components,
                                                  max_components=max_components,
                                                  criterion=criterion)
        c = column[-1]
        df.to_markdown(os.path.join(data_tools.DATA_DIR,
                                    f"mixture-{dataset}_{mode}_{c}.md"))
        print(f"{column}, best c = {mixture_model and mixture_model.n_components}")
        #print(df)


def examine_single():
    args = sys.argv
    if len(args) != 4:
        print("Usage:")
        print(f"    {args[0]} DATASET MODE aic|bic")
        return
    examine_dataset(args[1].upper(), args[2].lower(), args[3].lower())


def examine_all():
    args = sys.argv
    if len(args) != 2:
        print("Usage:")
        print(f"    {args[0]} aic|bic")
        return
    for mode in data_tools.MODES:
        for dataset in data_tools.DATASETS:
            examine_dataset(dataset, mode, criterion=args[1].lower())
            print("")


def main():
    examine_all()


if __name__ == "__main__":
    main()
