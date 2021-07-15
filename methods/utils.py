def zero_matrix(rows, cols):
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)
    return M


def identity_matrix(n):
    I = zero_matrix(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def copy_matrix(M):
    rows = len(M)
    cols = len(M[0])

    MC = zero_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC


def print_matrix(M, decimals=3):
    for row in M:
        print([round(x, decimals) + 0 for x in row])


def transpose(M):
    if not isinstance(M[0], list):
        M = [M]

    rows = len(M)
    cols = len(M[0])

    MT = zero_matrix(cols, rows)

    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT


def sum_matrices(A, B):
    if isinstance(A[0], list) and isinstance(B[0], list):
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise ArithmeticError('Las matrices deben tener el mismo tamaño para poder sumarse')

        C = zero_matrix(len(A), len(A[0]))
        for i in range(len(A)):
            for j in range(len(A[0])):
                C[i][j] = A[i][j] + B[i][j]
        return C
    elif not isinstance(A[0], list) and not isinstance(B[0], list):
        if len(A) != len(B):
            raise ArithmeticError('Las matrices deben tener el mismo tamaño para poder sumarse')

        C = zero_matrix(len(A), 1)
        for i in range(len(A)):
            C[i][1] = A[i] + B[i]
        return C
    else:
        if isinstance(A[0], list):
            for i in range(len(B)):
                B[i] = [B[i]]
        else:
            for i in range(len(A)):
                A[i] = [A[i]]
        return sum_matrices(A, B)


def subtract_matrices(A, B):
    if isinstance(A[0], list) and isinstance(B[0], list):
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise ArithmeticError('Las matrices deben tener el mismo tamaño para poder restarse')

        C = zero_matrix(len(A), len(A[0]))
        for i in range(len(A)):
            for j in range(len(A[0])):
                C[i][j] = A[i][j] - B[i][j]
        return C
    elif not isinstance(A[0], list) and not isinstance(B[0], list):
        if len(A) != len(B):
            raise ArithmeticError('Las matrices deben tener el mismo tamaño para poder restarse')

        C = zero_matrix(len(A), 1)
        for i in range(len(A)):
            C[i][1] = A[i] - B[i]
        return C
    else:
        if isinstance(A[0], list):
            for i in range(len(B)):
                B[i] = [B[i]]
        else:
            for i in range(len(A)):
                A[i] = [A[i]]
        return subtract_matrices(A, B)


def mult_escalar(M, lambda_):
    rows = len(M)
    cols = len(M[0])

    MC = copy_matrix(M)

    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j] * lambda_

    return MC


def multiply_matrices(A, B):
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    if colsA != rowsB:
        raise ArithmeticError('El numero de columnas de A debe ser igual al numero de filas de B')

    C = zero_matrix(rowsA, colsB)
    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total

    return C


def compare_matrices(A, B, tol=None):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False

    for i in range(len(A)):
        for j in range(len(A[0])):
            if tol is None:
                if A[i][j] != B[i][j]:
                    return False
            else:
                if round(A[i][j], tol) - round(B[i][j], tol) > 1:
                    return False
    return True


def add_column_to_matrix(column_vector, M, column_num):

    rows = len(M)
    cols = len(M[0])

    if not isinstance(column_vector, list):
        column_value = column_vector
        column_vector = []
        for i in range(rows):
            column_vector.append([column_value])

    if rows != len(column_vector):
        raise ArithmeticError('Las filas de la nueva colunma no son iguales a las de la matriz')

    for i in range(rows):
        M[i].insert(column_num, column_vector[i][0])

    return M


def is_square_matrix(A):
    if len(A) != len(A[0]):
        raise ArithmeticError("La matriz debe ser cuadrada para poder ser invertida")


def determinant(A):
    """Calcula el determiante con el metodo de la matriz triangular superior"""
    n = len(A)
    AM = copy_matrix(A)

    for fd in range(n):
        if AM[fd][fd] == 0:
            AM[fd][fd] = 1.0e-18
        for i in range(fd + 1, n):
            cr_scaler = AM[i][fd] / AM[fd][fd]
            for j in range(n):
                AM[i][j] = AM[i][j] - cr_scaler * AM[fd][j]

    product = 1.0
    for i in range(n):
        product *= AM[i][i]

    return product


def no_singular_matrix(A):
    det = determinant(A)
    if det != 0:
        return det
    else:
        raise ArithmeticError("Matriz Singular")

def solve_equations(A, B, tol=None):
    is_square_matrix(A)
    no_singular_matrix(A)

    n = len(A)
    AM = copy_matrix(A)
    I = identity_matrix(n)
    BM = copy_matrix(B)

    indices = list(range(n))
    for fd in range(n):
        if AM[fd][fd] == 0:
            AM[fd][fd] = 1.0e-18
        fd_scaler = 1.0 / AM[fd][fd]

        for j in range(n):
            AM[fd][j] *= fd_scaler
        BM[fd][0] *= fd_scaler

        for i in indices[0:fd] + indices[fd + 1:]:
            crScaler = AM[i][fd]
            for j in range(n):
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
            BM[i][0] = BM[i][0] - crScaler * BM[fd][0]

    if compare_matrices(B, multiply_matrices(A, BM), tol):
        return BM
    else:
        raise ArithmeticError("Solución para X fuera de tolerancia")