#!python3


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import mnist


if __name__ == '__main__':
    exp_disp = float(input("Доля объяснённой дисперсии: "))
    test_ratio = float(input("Доля тестовой выборки: "))
    random_state = int(input("random_state = "))
    min_samples_leaf = int(input("min_samples_leaf = "))
    max_depth = int(input("max_depth = "))
    n_estimators = int(input("n_estimators = "))
    clazz_randomforest = int(input("Класс для RandomForestClassifier: "))
    clazz_log_reg = int(input("Класс для LogisticRegression: "))
    clazz_tree = int(input("Класс для DecisionTreeClassifier: "))
    image_file_randomforest = input("Изображение для RandomForestClassifier: ")
    image_file_log_reg = input("Изображение для LogisticRegression: ")
    image_file_tree = input("Изображение для DecisionTreeClassifier: ")

    (X_train, y_train), (X_pred, y_pred) = mnist.load_data()

    dim = 784  # 28*28
    X_train_ = X_train.reshape(len(X_train), dim)

    pca = PCA(svd_solver='full')
    pca = pca.fit(X_train_)

    M = 0
    for arg, val in enumerate(np.cumsum(pca.explained_variance_ratio_)):
        if val > exp_disp:
            M = arg + 1
            break

    print("Количество главных компонент, чтобы доля объяснённой дисперсии превышала " + str(exp_disp) + ": " + str(M))

    X_train = X_train.reshape(len(X_train), dim)

    # Поиск счёт
    pca = PCA(n_components=M, svd_solver='full')
    pca = pca.fit(X_train)

    # Разделение выборки
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_ratio,
                                                        random_state=random_state)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    print("Выборочное среднее нулевой колонки тренировочного набора: " + str(X_train.transpose()[0].mean()))

    random_forest = RandomForestClassifier(criterion='gini',
                                           min_samples_leaf=min_samples_leaf,
                                           max_depth=max_depth,
                                           n_estimators=n_estimators,
                                           random_state=random_state)
    clf_random_forest = OneVsRestClassifier(random_forest).fit(X_train, y_train)
    y_pred = clf_random_forest.predict(X_test)

    CM = confusion_matrix(y_test, y_pred)
    print("Количество верно классифицированных объектов класса " + str(clazz_randomforest) + ": " +
          str(CM[clazz_randomforest][clazz_randomforest]))

    log_reg = LogisticRegression(random_state=random_state, solver='lbfgs')
    clf_log_reg = OneVsRestClassifier(log_reg).fit(X_train, y_train)
    y_pred = clf_log_reg.predict(X_test)

    CM = confusion_matrix(y_test, y_pred)
    print("Количество верно классифицированных объектов класса " + str(clazz_log_reg) + ": " +
          str(CM[clazz_log_reg][clazz_log_reg]))

    tree = DecisionTreeClassifier(criterion='gini',
                                  min_samples_leaf=min_samples_leaf,
                                  max_depth=max_depth,
                                  random_state=random_state)
    clf_tree = OneVsRestClassifier(tree).fit(X_train, y_train)
    y_pred = clf_tree.predict(X_test)

    CM = confusion_matrix(y_test, y_pred)
    print("Количество верно классифицированных объектов класса " + str(clazz_tree) + ": " +
          str(CM[clazz_tree][clazz_tree]))


    DATA = pd.read_csv("pred_for_task.csv", delimiter=',', index_col='FileName')
    X_test = pd.DataFrame(DATA.drop(['Label'], axis=1))
    X_test = pca.transform(X_test)

    y_pred = clf_random_forest.predict_proba(X_test)
    idx = list(DATA.index).index(image_file_randomforest)
    print("Вероятность отнесения изображения " + image_file_randomforest + " к назначенному классу: " +
          str(y_pred[idx][DATA['Label'][idx]]))

    y_pred = clf_log_reg.predict_proba(X_test)
    idx = list(DATA.index).index(image_file_log_reg)
    print("Вероятность отнесения изображения " + image_file_log_reg + " к назначенному классу: " +
          str(y_pred[idx][DATA['Label'][idx]]))

    y_pred = clf_tree.predict_proba(X_test)
    idx = list(DATA.index).index(image_file_tree)
    print("Вероятность отнесения изображения " + image_file_tree + " к назначенному классу: " +
          str(y_pred[idx][DATA['Label'][idx]]))
