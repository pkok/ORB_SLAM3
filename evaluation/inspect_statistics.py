import numpy as np
import pandas
import scipy.stats
import math

from evaluate_ate_scale import normalize_data

MODES = ["mono", "stereo", "monoi", "stereoi"]

DATASETS = ["MH01", "MH02", "MH03", "MH04", "MH05",
            "V101", "V102", "V103",
            "V201", "V202", "V203"]

def ks_rejection_at_alpha(D, n, m):
    return 2 * math.exp(-(2 * D * D * m * n)/(m + n))

def ks_rejection_threshold(alpha, n, m):
    return math.sqrt(-math.log(alpha/2) * (1 + (m/n)) / (2*m))

def compare(data, mode, dataset, prefix=''):
    #columns = []
    #for c in data.columns:
    #    #columns.append(c + " U")
    #    #columns.append(c + " p")
    #    columns.append(c)
    #mann_whitneys = pandas.DataFrame(columns=columns,
    #                                 index=data.columns)
    #columns = []
    #for c in data.columns:
    #    #columns.append(c + " T")
    #    #columns.append(c + " p")
    #    columns.append(c)
    #wilcoxons = pandas.DataFrame(columns=columns,
    #                             index=data.columns)
    columns = data.columns
    ks = pandas.DataFrame(columns=columns,
                          index=data.columns)

    for c1 in data.columns:
        for c2 in data.columns:
            U, p = scipy.stats.mannwhitneyu(data[c1],
                                            data[c2],
                                            alternative='two-sided')
            ##mann_whitneys[c1+" U"][c2] = U
            ##mann_whitneys[c1][c2+" p"] = p
            #mann_whitneys[c1][c2] = p

            ##T, p = scipy.stats.ranksums(data[c1],
            ##                            data[c2])
            ##wilcoxons[c1+" T"][c2] = T
            ##wilcoxons[c1+" p"][c2] = p
            #wilcoxons[c1][c2] = p

            D, p = scipy.stats.kstest(data[c1],
                                      data[c2])
            ks[c1][c2] = ks_rejection_at_alpha(D,
                                               len(data[c1]),
                                               len(data[c2]))
            #ks[c1][c2] = int(D <= ks_rejection_threshold(0.001,
            #                                             len(data[c1]),
            #                                             len(data[c2])))

    #mann_whitneys.to_csv(f"../data/{prefix}mannwhitney-{dataset}_{mode}.csv")
    #mann_whitneys.to_excel(f"../data/{prefix}mannwhitney-{dataset}_{mode}.xlsx")
    #mann_whitneys.to_markdown(f"../data/{prefix}mannwhitney-{dataset}_{mode}.md")
    #mann_whitneys.to_latex(f"../data/{prefix}mannwhitney-{dataset}_{mode}.tex")

    #wilcoxons.to_csv(f"../data/{prefix}wilcoxon-{dataset}_{mode}.csv")
    #wilcoxons.to_excel(f"../data/{prefix}wilcoxon-{dataset}_{mode}.xlsx")
    #wilcoxons.to_markdown(f"../data/{prefix}wilcoxon-{dataset}_{mode}.md")
    #wilcoxons.to_latex(f"../data/{prefix}wilcoxon-{dataset}_{mode}.tex")

    ks.to_excel(f"../data/{prefix}kstest-{dataset}_{mode}.xlsx")

if __name__ == "__main__":
    for mode in MODES:
        for dataset in DATASETS:
            data = pandas.read_csv(f"../data/errors-{dataset}_{mode}.csv")
            compare(data, mode, dataset)
            compare(normalize_data(data), mode, dataset,
                    prefix='normalized-')
