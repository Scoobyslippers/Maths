import numpy as np
#from scipy.sparse import diags


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

    # ...
    L[0, 0] = 1
    U[0, 0] = 1

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

    for rows in range(1,n): ##RUns from row 1 (ALREADY CONFIRMED FIRST VALUE) to the final row
        for columns in range(rows,n): ##goes from the current value of row column to the final column - so we arent wriritng on the values that are 0
            SumValue = np.dot(L[rows, :rows], U[:rows, columns]) #Ths creates the sum value so they can subtract - this is the condensed sum when we take out the 0s
            U[rows, columns] = A[rows, columns] - SumValue #Substracts the sum value that is created from the current value in A
        for columns in range(rows+1,n): #RUns from the next row value, to the final column value
            sumNewVal = np.dot(L[columns, :rows], U[:rows, rows]) #Same as the sum val above then
            L[columns, rows] = (A[columns, rows]-sumNewVal) / U[rows, rows] ##Finishes the value of L by subtracting the Sum val ten dividing it by the row value
    
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


A_test = np.array([[4, 3, 2, 1],
                   [3, 3, 2, 1],
                   [2, 2, 2, 1],
                   [1, 1, 1, 1]], dtype=float)

try:
    L, U = lu_factorisation(A_test)
    print("Matrix L:\n", L)
    print("\nMatrix U:\n", U)
    print(determinant(A_test))

except ValueError as e:
    print(f"Error: {e}")