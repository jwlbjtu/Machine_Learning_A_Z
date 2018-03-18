
**Numpy** is a popular Python libaray in Machine Learning area. Here summarised some useful tips for Numpy.


```python
import numpy as np
```

### Basics of Numpy Array 


```python
# creating numpy array
array = np.array([[1, 2, 3],
                  [2, 3, 4]])

print(array)
print('Number of dim:', array.ndim)
print('Shape:', array.shape)
print('size:', array.size)
```

    [[1 2 3]
     [2 3 4]]
    Number of dim: 2
    Shape: (2, 3)
    size: 6



```python
# define element type of an array
array = np.array([1, 2, 3], dtype = np.int)
print(array.dtype) # int64

array = np.array([1, 2, 3], dtype = np.int32)
print(array.dtype) # int32

array = np.array([1, 2, 3], dtype = np.float)
print(array.dtype) # float64

array = np.array([1, 2, 3], dtype = np.float32)
print(array.dtype) # float32
```

    int64
    int32
    float64
    float32


### Creating Matrix


```python
# creating a matrix with all 0
matrix = np.zeros((2,3))
print('Zeros matrix')
print(matrix)

# creating a matrix with all 1
matrix = np.ones((3,4))
print('Ones matrix')
print(matrix)

# creating empty matrix
matrix = np.empty((2,2))
print('Empty matrix')
print(matrix)

# creating random matrix
matrix = np.random.random((2,4))
print('Random matrix')
print(matrix)
```

    Zeros matrix
    [[ 0.  0.  0.]
     [ 0.  0.  0.]]
    Ones matrix
    [[ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]]
    Empty matrix
    [[  2.68156159e+154  -2.32036126e+077]
     [  6.94773593e-310   2.78136381e-309]]
    Random matrix
    [[ 0.3361672   0.15099262  0.43580346  0.07635681]
     [ 0.21574756  0.40070359  0.64789344  0.66923312]]


### Numpy Arange


```python
# creating continuous array
a = np.arange(10, 20, 2)
print(a)
```

    [10 12 14 16 18]



```python
# creating continuous matrix
a = np.arange(12).reshape((3,4))
print(a)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]



```python
# creating a linspace
a = np.linspace(1, 10, 5)
print(a)

print()

a = np.linspace(1, 10, 6).reshape((2, 3))
print(a)
```

    [  1.     3.25   5.5    7.75  10.  ]
    
    [[  1.    2.8   4.6]
     [  6.4   8.2  10. ]]


### Numpy Operations 


```python
# array operations
a = np.array([10, 20, 30, 40])
b = np.arange(4)
print('a = ', a)
print('b = ', b)

c = a - b
print('Substraction:', c)

c = a + b
print('Sum:', c)

c = b**2
print('Square:', c)

c = 10 * np.sin(a)
print('Sin:', c)

c = 10 * np.cos(a)
print('Cos:', c)

c = 10 * np.tan(a)
print('Tan:', c)
```

    a =  [10 20 30 40]
    b =  [0 1 2 3]
    Substraction: [10 19 28 37]
    Sum: [10 21 32 43]
    Square: [0 1 4 9]
    Sin: [-5.44021111  9.12945251 -9.88031624  7.4511316 ]
    Cos: [-8.39071529  4.08082062  1.5425145  -6.66938062]
    Tan: [  6.48360827  22.37160944 -64.05331197 -11.17214931]



```python
# array element examination
print(b)
print(b < 3)
```

    [0 1 2 3]
    [ True  True  True False]



```python
# matrix operation
a = np.array([[1, 1],
              [0, 1]])
b = np.arange(4).reshape((2,2))

print('a = ', a)
print('b = ', b)

# element multiply
c = a * b 
print('Element Multiply:')
print(c)

# matrix multiply
c = np.dot(a, b)
print('Matrix Multiply:')
print(c)

c = a.dot(b)
print('Matrix Multiply:')
print(c)
```

    a =  [[1 1]
     [0 1]]
    b =  [[0 1]
     [2 3]]
    Element Multiply:
    [[0 1]
     [0 3]]
    Matrix Multiply:
    [[2 4]
     [2 3]]
    Matrix Multiply:
    [[2 4]
     [2 3]]



```python
# matrix internal operations
a = np.random.random((2,4))
print('a = ', a)

# mean of the matrix
print('mean:', np.mean(a)) 
print('mean:', a.mean())

# average of the matrix
print('average:', np.average(a))

# median of the matrix
print('median:', np.median(a)) 

# sum of the matrix
print('sum:', np.sum(a)) 

# min element of the matrix
print('min:', np.min(a))

# max element of the matrix
print('max:', np.max(a))

# sum of each row in the matrix
print('sum of row:', np.sum(a, axis = 1))  

# min element of each column in the matrix
print('min of column:', np.min(a, axis = 0))

# max element of each row in the matrix
print('max of row:', np.max(a, axis =1))

# cumulative sum
print('cumsum:', np.cumsum(a))

# diff between elements
print('diff:', np.diff(a))

# sort the matrix
print('sort:', np.sort(a))

# matrix transpose
print('trnaspose:')
print(np.transpose(a))
print('trnaspose:')
print(a.T)

# flatten a matrix
print('flatten matrix:', a.flatten())
```

    a =  [[ 0.81709816  0.84494671  0.48110455  0.15578455]
     [ 0.26020247  0.20752249  0.93577815  0.10599093]]
    mean: 0.476053500013
    mean: 0.476053500013
    average: 0.476053500013
    median: 0.370653509283
    sum: 3.8084280001
    min: 0.10599093105
    max: 0.935778148161
    sum of row: [ 2.29893396  1.50949404]
    min of column: [ 0.26020247  0.20752249  0.48110455  0.10599093]
    max of row: [ 0.84494671  0.93577815]
    cumsum: [ 0.81709816  1.66204487  2.14314942  2.29893396  2.55913643  2.76665892
      3.70243707  3.808428  ]
    diff: [[ 0.02784855 -0.36384216 -0.32532   ]
     [-0.05267998  0.72825566 -0.82978722]]
    sort: [[ 0.15578455  0.48110455  0.81709816  0.84494671]
     [ 0.10599093  0.20752249  0.26020247  0.93577815]]
    trnaspose:
    [[ 0.81709816  0.26020247]
     [ 0.84494671  0.20752249]
     [ 0.48110455  0.93577815]
     [ 0.15578455  0.10599093]]
    trnaspose:
    [[ 0.81709816  0.26020247]
     [ 0.84494671  0.20752249]
     [ 0.48110455  0.93577815]
     [ 0.15578455  0.10599093]]
    flatten matrix: [ 0.81709816  0.84494671  0.48110455  0.15578455  0.26020247  0.20752249
      0.93577815  0.10599093]



```python
# indexing in matrix
A = np.arange(2, 14).reshape((3, 4))
print('A = ')
print(A)

# min index
print('min index', np.argmin(A))

# max index
print('max index', np.argmax(A))

# find non-zero index
print('non-zero', np.nonzero(A))

# clip
print('clip')
print(np.clip(A, 5, 9))
```

    A = 
    [[ 2  3  4  5]
     [ 6  7  8  9]
     [10 11 12 13]]
    min index 0
    max index 11
    non-zero (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]))
    clip
    [[5 5 5 5]
     [6 7 8 9]
     [9 9 9 9]]


### Numpy Index 


```python
# array index
A = np.arange(3, 15)
print('A = ')
print(A)

# find element with index
print('Element with index 3:', A[3])
```

    A = 
    [ 3  4  5  6  7  8  9 10 11 12 13 14]
    Element with index 3: 6



```python
# matrix index
A = np.arange(3, 15).reshape((3, 4))
print('A = ')
print(A)

# find row
print('Row with index 2:', A[2])
print('Row with index 2:', A[2, :])
print('Elment index between 1 and 3(exclude) in row with index 1:', A[1, 1:3])

# find column elements
print('Column with index 2:', A[:, 2])

# find element
print('Element at row index 2 nad column index 1:', A[2][1])
print('Element at row index 2 nad column index 1:', A[2, 1])
```

    A = 
    [[ 3  4  5  6]
     [ 7  8  9 10]
     [11 12 13 14]]
    Row with index 2: [11 12 13 14]
    Row with index 2: [11 12 13 14]
    Elment index between 1 and 3(exclude) in row with index 1: [8 9]
    Column with index 2: [ 5  9 13]
    Element at row index 2 nad column index 1: 12
    Element at row index 2 nad column index 1: 12


### Iterate Matrix 


```python
# iteration of matrix
i = 0
for row in A:
    print('row no.', i, row)
    i += 1
```

    row no. 0 [3 4 5 6]
    row no. 1 [ 7  8  9 10]
    row no. 2 [11 12 13 14]



```python
i = 0
for column in A.T:
    print('colomun no.', i, column)
    i += 1
```

    colomun no. 0 [ 3  7 11]
    colomun no. 1 [ 4  8 12]
    colomun no. 2 [ 5  9 13]
    colomun no. 3 [ 6 10 14]



```python
for item in A.flat:
    print(item)
```

    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14


### Merage  


```python
A = np.array([1, 1, 1])
B = np.array([2, 2, 2])
print('A =', A)
print('B =', B)

# vertical stack
print('vertical stack:')
print(np.vstack((A, B)))

# horizontal stack
print('horizontal stack:')
print(np.hstack((A, B)))

# add dimension
print('add row dimension:', A[np.newaxis, :])
print('add column dimension:')
print(A[:, np.newaxis])

# concatenate
A = A[:, np.newaxis]
B = B[:, np.newaxis]
C = np.concatenate((A, B, B, A), axis = 1)
print('concatenate:')
print(C)
```

    A = [1 1 1]
    B = [2 2 2]
    vertical stack:
    [[1 1 1]
     [2 2 2]]
    horizontal stack:
    [1 1 1 2 2 2]
    add row dimension: [[1 1 1]]
    add column dimension:
    [[1]
     [1]
     [1]]
    concatenate:
    [[1 2 2 1]
     [1 2 2 1]
     [1 2 2 1]]


### Divide 


```python
A = np.arange(12).reshape((3, 4))
print('A =')
print(A)

# horizontal divide
print('horizontal divide')
print(np.split(A, 2, axis = 1))
print(np.hsplit(A, 2))

# vertical divide
print('vertical divide')
print(np.split(A, 3, axis = 0))
print(np.vsplit(A, 3))

# non-even divide
print('non-even divide')
print(np.array_split(A, 3, axis = 1))
```

    A =
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    horizontal divide
    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2,  3],
           [ 6,  7],
           [10, 11]])]
    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2,  3],
           [ 6,  7],
           [10, 11]])]
    vertical divide
    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
    non-even divide
    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2],
           [ 6],
           [10]]), array([[ 3],
           [ 7],
           [11]])]


### Copy


```python
a = np.arange(4)
print('a:', a)

b = a 
c = np.copy(a)
a[1:3] = [100, 111]

print('a:', a)
print('b:', b)
print('c:', c)
```

    a: [0 1 2 3]
    a: [  0 100 111   3]
    b: [  0 100 111   3]
    c: [0 1 2 3]

