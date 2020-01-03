import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data.futures import Futures

h5_data_file = "data.h5"


def load(code, window, force=False) -> pd.DataFrame:
    if force:
        return _force_load(code, window)
    try:
        df = pd.DataFrame(pd.read_hdf(h5_data_file, str(code) + "_concat_" + str(window)))
        if df.empty:
            return _force_load(code, window)
        return df
    except (FileNotFoundError, KeyError):
        return _force_load(code, window)


def _force_load(code, window):
    df = Futures.get(code)
    if not df.empty:
        df.to_hdf(h5_data_file, code)
    else:
        return df
    windows = _rolling(df, window)
    new_windows = []
    for w in windows:
        new_windows.append(_concat(w))
    df = pd.concat(new_windows)
    df.to_hdf(h5_data_file, str(code) + "_concat_" + str(window))
    return df


def _rolling(df: pd.DataFrame, window: int):
    for i in range(len(df) - window):
        yield df.iloc[i:i + window, :]


def _concat(df: pd.DataFrame):
    index = df.index[0]
    groups = df.groupby(level=0)
    new_groups = []
    for i, gt in enumerate(groups):
        g = gt[1]
        cols = [c + "_" + str(i + 1) if i != len(df) - 1 else c for c in g.columns]
        g.columns = cols
        g.index = [index]
        new_groups.append(g)
    return pd.concat(new_groups, axis=1, ignore_index=False)


def main():
    df = load("a0", 6)
    df["target"] = (df["close"] - df["close_5"]).map(lambda x: 1 if x > 0 else 0)
    scaler = StandardScaler()
    for c in df.columns:
        if c == "target":
            continue
        df[c] = scaler.fit_transform(df[[c]])
    train = df.iloc[:-300, :]
    test = df.iloc[-300:, :]
    X = train.iloc[:, :-6]
    y = train.iloc[:, [-1]]
    xgb = XGBRegressor(n_estimators=80, max_depth=2, learning_rate=0.03, objective="reg:squarederror")
    test_params = {
        "n_estimators": range(80, 500, 5),
        "max_depth": range(2, 7, 1)
    }
    cv = GridSearchCV(xgb, param_grid=test_params, n_jobs=-1)
    cv.fit(X, y)
    print(cv.best_params_, cv.best_score_)


if __name__ == '__main__':
    main()
