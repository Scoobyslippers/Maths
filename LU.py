import numpy as np
from scipy.sparse import diags

def generate_safe_system(n):
    """
    Generate a linear system A x = b where A is strictly diagonally dominant,
    ensuring LU factorization without pivoting will work.

    Parameters:
        n (int): Size of the system (n x n)

    Returns:
        A (ndarray): n x n strictly diagonally dominant matrix
        b (ndarray): RHS vector
        x_true (ndarray): The true solution vector
    """

    k = [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)]
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()

    # Solution is always all ones
    x_true = np.ones((n, 1))

    # Compute b = A @ x_true
    b = A @ x_true

    return A, b, x_true

def lu_factorisation(A):
    """
    Compute the LU factorisation of a square matrix A.

    The function decomposes a square matrix ``A`` into the product of a lower
    triangular matrix ``L`` and an upper triangular matrix ``U`` such that:

    .. math::
        A = L U

    where ``L`` has unit diagonal elements and ``U`` is upper triangular.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the square matrix to
        factorise.

    Returns
    -------
    L : numpy.ndarray
        A lower triangular matrix with shape ``(n, n)`` and unit diagonal.
    U : numpy.ndarray
        An upper triangular matrix with shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    # construct arrays of zeros
    L, U = np.zeros_like(A), np.zeros_like(A)
    L = L = np.eye(n, dtype=float) #Make Diagonal Matrix to 1s

    #Create the Diagonal matrix (D)
    i = 0
    for i in range(n):
        L[i,i] = 1 ##Creates the Diagonal matrix
    B = A.copy()
##This is the foor loop will be for L
    ##Start by getting the top row of U

    for columns in range(n):
        U[0, columns] = A[0, columns] ##This sets the value of the first row in U to the exact same value as A[]
        ## E.G. A = [ 4 2 0 ] ---> U = [4 2 0]
        ##          [ x x x ]          [0 X X]
        ##          [ x x x ]          [0 0 X]

    for rows in range(n): #Goes through the rows in the first column
        if rows >= 1 : #Ignores the first element as its set to 1 - diag matrix
            divisor = U[0, 0] #Sets the value of the div to the first value in U, this case is 4
            L[rows,0] = A[rows, 0 ] / divisor #This divides the first row values of A, by the value of the divisor  
        ## E.G U = [4 X X] A = [4 X X] --> L = [1   0 0]
        ##         [X X X]     [2 X X]         [2/4 0 0]
        ##         [X X X]     [0 X X]         [0/4 0 0]
        for i in range(n):
            for j in range(i,n):
                sumValue = np.dot(L[i, :i], U[:i, j])
                U[i,j] = B[i,j] - sumValue
            for j in range(i+1,n):
                sumValue = np.dot(L[i, :i], U[:i, j])

                if np.isclose(U[i,i],0):
                    raise ValueError("Zero pivot present, LU without pivoting failed")
                L[j, i] = (A[j, i] - sumValue) / U[i, i]

    return L,U

def determinant(A):
    n = A.shape[0]
    L, U = lu_factorisation(A)

    detL = 1
    detU = 1

    for i in range(n):
        detL *= L[i, i]
        detU *= U[i, i]
    
    return detL * detU


A_test = np.array([[2, 3, 4],
                   [5, 6, 7],
                   [1, 2, 3]], dtype=float)

try:
    L, U = lu_factorisation(A_test)
    print("Matrix L:\n", L)
    print("\nMatrix U:\n", U)
    print(determinant(A_test))

    ALarge, BLarge, XLarge = generate_safe_system(100)
    print(ALarge)
    print (determinant(ALarge))
except ValueError as e:
    print(f"Error: {e}")