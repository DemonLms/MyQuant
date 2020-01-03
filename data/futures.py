import pandas as pd
import requests


class Futures:
    @staticmethod
    def get(code) -> pd.DataFrame:
        code = str(code).upper()
        url = "https://stock.sina.com.cn/futures/api/json.php/InnerFuturesNewService.getDailyKLine?symbol={}".format(
            code)
        res = requests.get(url)
        if res.status_code != 200:
            raise ConnectionError(res.status_code)
        df = pd.read_json(res.text)
        df["d"] = pd.to_datetime(df["d"])
        resort = ["d", "o", "h", "l", "c", "v"]
        df = df.loc[:, resort]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df = df.set_index("date")
        return df


if __name__ == '__main__':
    d = Futures.get("a0")
    pd.DataFrame().to_hdf("data.h5", "a0")
    print(d)
