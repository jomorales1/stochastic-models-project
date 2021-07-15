from utils import transpose, multiply_matrices, identity_matrix
from utils import mult_escalar, sum_matrices, solve_equations, subtract_matrices


class RidgeRegression:
    def ridge_solution(self, X, Y, lambda_, tol=3):
        if not isinstance(X[0], list):
            X = [X]
        if not isinstance(type(Y[0]), list):
            Y = [Y]

        if len(X) < len(X[0]):
            X = transpose(X)
        if len(Y) < len(Y[0]):
            Y = transpose(Y)

        for i in range(len(X)):
            X[i].append(1)

        AT = transpose(X)
        ATA = multiply_matrices(AT, X)
        I = identity_matrix(len(ATA))
        I = mult_escalar(I, lambda_)
        ATA = sum_matrices(ATA, I)
        ATB = multiply_matrices(AT, Y)
        coefs = solve_equations(ATA, ATB, tol=tol)

        return coefs

    def ridge_rss(self, X, Y, lambda_):
        coef = self.ridge_solution(X, Y, lambda_)
        rss = multiply_matrices(transpose(subtract_matrices(Y, multiply_matrices(X, coef))),
                                 subtract_matrices(Y, multiply_matrices(X, coef)))
        if not isinstance(rss, list):
            return rss
        elif not isinstance(rss[0], list):
            return rss[0]
        else:
            return rss[0][0]
