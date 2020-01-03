from itertools import product

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
from data.futures import Futures


def coint(name1, name2):
    c1 = Futures.get(name1)["close"].rename("c1")
    c2 = Futures.get(name2)["close"].rename("c2")

    tmp = pd.concat([c1, c2], axis=1, join="inner")
    c1 = tmp["c1"]
    c2 = tmp["c2"]
    corr_pvalue = np.corrcoef(c1, c2)[0][1]
    if corr_pvalue < 0.75:
        return
    pvalue = sm.tsa.stattools.coint(c1, c2)[1]
    if pvalue > 0.2:
        return
    X = sm.add_constant(c1.to_numpy())
    result = sm.OLS(c2, X).fit()
    c = c2 - c1 * result.params[1] - result.params[0]
    std = np.std(c)
    mean = np.mean(c)

    plt.subplot(211)
    plt.title("%s-%.2f*%s-%.2f" % (name2, result.params[1], name1, result.params[0]))
    plt.plot(c)
    plt.plot(pd.Series(mean, index=c.index))
    plt.plot(pd.Series(mean + std, index=c.index))
    plt.plot(pd.Series(mean - std, index=c.index))

    c = c2 - c1
    std = np.std(c)
    mean = np.mean(c)

    plt.subplot(212)
    plt.title("%s-%s" % (name2, name1))
    plt.plot(c)
    plt.plot(pd.Series(mean, index=c.index))
    plt.plot(pd.Series(mean + std, index=c.index))
    plt.plot(pd.Series(mean - std, index=c.index))
    plt.show()
    return name1, name2


def hedge():
    months = ["2005", "2009"]
    pairs = [("c", "cs"), ("c", "a"), ("a", "m"), ("rm", "m")]
    trade_pairs = [tuple(map(lambda pp: pp + ym, p)) for p, ym in
                   product(pairs, months)]
    for tp in trade_pairs:
        x = coint(*tp)
        if x is not None:
            print(x)


def calendar():
    month_pairs = [('2005', '2009')]
    names = ["c", "cs", "a", "m", "rm"]
    trade_pairs = [tuple(map(lambda pp: yn + pp, p)) for yn, p in
                   product(names, month_pairs)]

    for tp in trade_pairs:
        x = coint(*tp)
        if x is not None:
            print(x)


if __name__ == '__main__':
    hedge()
    calendar()
    # coint("cs2005","cs2009")
