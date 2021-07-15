from utils import multiply_matrices, transpose, solve_equations


class OrdinaryLeastSquares:
    def predict(self, X, Y, tol=3):
        
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
        ATB = multiply_matrices(AT, Y)
        coefs = solve_equations(ATA, ATB, tol=tol)

        return coefs
