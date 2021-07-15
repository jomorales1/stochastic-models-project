from methods.ordinary_least_squares import OrdinaryLeastSquares
from utils import copy_matrix, multiply_matrices, transpose, subtract_matrices


class BestSubsetBSS:
    def __RSS(self, X, Y, tol=3):
        coef = OrdinaryLeastSquares.predict(X, Y, tol)
        rss = multiply_matrices(
            transpose(
                subtract_matrices(
                    Y,
                    multiply_matrices(X, coef)
                )
            ),
            subtract_matrices(
                Y,
                multiply_matrices(X, coef)
            )
        )
        if not isinstance(rss, list):
            return rss
        elif not isinstance(rss[0], list):
            return rss[0]
        else:
            return rss[0][0]

    def predict(self, X, Y, k):
        subset = (X, Y)
        subset_len = len(subset[1])
        while subset_len > k:
            minRSS = 1e10
            minX = []
            minY = []
            for i in range(subset_len):
                x = copy_matrix(subset[0])
                x.pop(i)
                y = subset[1].copy()
                y.pop(i)

                rss = self.__RSS(copy_matrix(x), y.copy())
                if (rss < minRSS):
                    minRSS = rss
                    minX = x
                    minY = y
            subset = (minX, minY, minRSS)
            subset_len = len(subset[1])
        return subset
