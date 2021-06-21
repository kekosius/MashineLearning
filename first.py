#!python3

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import DistanceMetric
import pandas as pd
import numpy as np


if __name__ == '__main__':
    target0_min, target0_max = tuple(eval(input('Минимальное и максимальное значения при TARGET = 0 в виде массива: ')))
    target1_min, target1_max = tuple(eval(input('Минимальное и максимальное значения при TARGET = 1 в виде массива: ')))
    star = eval(input('Звезда в виде массива: '))

    DATA = pd.read_csv("pulsar_stars_new.csv", delimiter=',')
    DATA = DATA[((DATA.TARGET == 0) & (DATA.MIP >= target0_min) & (DATA.MIP <= target0_max))
                | ((DATA.TARGET == 1) & (DATA.MIP >= target1_min) & (DATA.MIP <= target1_max))]

    print("Число строк: " + str(len(DATA)))
    print("Выборочое среднее MIP до нормировки: " + str(DATA.MIP.mean()))

    scaler = MinMaxScaler()

    scaler.fit(DATA)
    DATA = pd.DataFrame(scaler.transform(DATA), columns=DATA.columns)

    print("Выборочое среднее MIP после нормировки: " + str(DATA.MIP.mean()))

    X = pd.DataFrame(DATA.drop(['TARGET'], axis=1))
    y = pd.DataFrame(DATA['TARGET'])

    reg = LogisticRegression(random_state=2019, solver='lbfgs').fit(X, y.values.ravel())
    for i, p in enumerate(reg.predict_proba([star])[0]):
        print("Вероятность отнесения к классу \"" + ("не пульсар" if i == 0 else "пульсар") + "\": " + str(p))

    for m in ["euclidean", "manhattan"]:
        dist = DistanceMetric.get_metric(m)
        distances = [dist.pairwise(np.concatenate(([i], [star])))[0][1] for i in X.values.tolist()]

        print("Расстояние до ближайшего по метрике \"" + m + "\": " + str(min(distances)))
