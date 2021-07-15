import numpy as np

"""
Clase que implementa un modelo de regresión lineal usando el algoritmo de ángulos mínimos,
descrito en http://statweb.stanford.edu/~tibs/ftp/lars.pdf.
"""


class LAR:
    def __init__(self, n_features):
        self.n_features = n_features
        self.coef = np.zeros(n_features)

    def __preprocess(self, X, y):
        return self.coef

    def __set_intercept(self, X_offset, X_scale, y_offset):
        self.coef /= X_scale
        self.intercept = y_offset - np.dot(X_offset, self.coef.T)

    def __predict(self, X):
        return np.dot(X, self.coef.T) + self.intercept

    def __min_pos(self, array):
        if not array.size:
            return 0
        min_value = float("inf")
        for e in array:
            if e > 0:
                min_value = min(min_value, e)
        return min_value

    def fit(self, X, y):
        assert isinstance(X, np.ndarray), "X debe ser un arreglo de Numpy (ndarray)."
        assert isinstance(y, np.ndarray), "y debe ser un arreglo de Numpy (ndarray)."
        assert X.shape[0] == y.shape[0], "X y Y deben tener el mismo número de ejemplos (filas)."
        self.n_features = X.shape[1]
        self.coef = np.zeros(self.n_features)
        Gram = np.dot(X.T, X)
        X, y, X_offset, X_scale, y_offset = self.__preprocess(X, y)
        A = []
        A_c = list(range(self.n_features))
        tiny32 = np.finfo(np.float32).tiny  # Para evitar división por zero.
        sign = np.zeros(self.n_features)
        for i in range(self.n_features):
            if i == 0:
                corr = np.dot(X.T, y)
            else:
                corr = np.dot(X.T, np.dot(X, self.coef.T)).reshape(self.n_features)
        C = np.max(np.abs(corr[A_c]))
        j = np.where(np.abs(corr) == C)[0][0]
        A.append(j)
        A_c.remove(j)
        sign = np.sign(corr[A])
        X_A = np.multiply(sign[:i + 1], X[:, A])
        g_A = np.dot(X_A.T, X_A)
        ones = np.ones((len(A), 1))
        A_A = 1. / np.sqrt(np.sum(np.linalg.pinv(g_A)))
        w_A = np.sum(A_A * np.linalg.pinv(np.dot(X_A.T, X_A)), axis=0)
        u_A = np.dot(X_A, w_A)
        a = np.dot(X.T, u_A)
        if i == self.n_features - 1:
            gamma = C / A_A
        else:
            gamma = min(self.__min_pos((C - corr[A_c]) / (A_A - a[A_c] + tiny32)),
                        self.__min_pos((C + corr[A_c]) / (A_A + a[A_c] + tiny32)))
        self.coef[A] += gamma * w_A.reshape((len(A),))
        self.__set_intercept(X_offset, X_scale, y_offset)
        return self.coef
