
Data preprocessing is one of the most critical step event before start thinking or using any machine learning models. A good data preprocessing can greatly improve the performence of the models. One another hand, if data is not prepared properly then the result of any model could be just "Garbage in Garbage out". 

Below are the typical steps to process a dataset:
1. Load the dataset, in order to get a sense of the data
2. Taking care of missing data (Optional)
3. Encoding categorical data (Optional)
4. Splitting dataset into the Training set and Test set (Validation set)
5. Feature Scaling

Thanks for all the powerful libararies, today we can implement above steps very easily with Python. 

## Import the libararies


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

- **[numpy](http://www.numpy.org/)** is a popular libaray for sintific computing. Here will mainly use it's N-dimensional array object. It also has very useful linear algebra, Fourier transform, and random number capabilities
- **[matploylib](https://matplotlib.org/)** is a Python 2D plotting library which can help us to visulize the dataset 
- **[pandas](https://pandas.pydata.org/)** is a easy-to-use data structures and data analysis tools for Python. We use it to load and separat datasets.
- **[sklearn](http://scikit-learn.org/stable/)** is another libaray we will use later. It is a very powerful tool for data analysis. Due to its comprehensive tools we will introduce them indivdualy once we use them. 

## Import the dataset


```python
# read a csv file by pandas
dataset = pd.read_csv('Data.csv')
# print out the loaded dataset
dataset
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Age</th>
      <th>Salary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>France</td>
      <td>44.0</td>
      <td>72000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>27.0</td>
      <td>48000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>30.0</td>
      <td>54000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spain</td>
      <td>38.0</td>
      <td>61000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>France</td>
      <td>35.0</td>
      <td>58000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spain</td>
      <td>NaN</td>
      <td>52000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>France</td>
      <td>48.0</td>
      <td>79000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>50.0</td>
      <td>83000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>France</td>
      <td>37.0</td>
      <td>67000.0</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
# separate the dataset into X and y
# X is independent variables. Here are columns 'Country', 'Age' and 'Salary'
X = dataset.iloc[:, :-1].values
# y is dependent variables. Here is column 'Purchased'
y = dataset.iloc[:, -1].values
```


```python
# value of X
print(X)
```

    [['France' 44.0 72000.0]
     ['Spain' 27.0 48000.0]
     ['Germany' 30.0 54000.0]
     ['Spain' 38.0 61000.0]
     ['Germany' 40.0 nan]
     ['France' 35.0 58000.0]
     ['Spain' nan 52000.0]
     ['France' 48.0 79000.0]
     ['Germany' 50.0 83000.0]
     ['France' 37.0 67000.0]]



```python
# value of y
print(y)
```

    ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']


## Taking care of missing data

If you look closely there are two missing values in the dataset. One is the age of customer 6. Another is the salary of customer 4. Most of the time we need to fullfill the missing values to make the model work. There are three main ways to do it. Using the 'mean', 'median' or 'most frequent'. Here I will implement by using the meam of each value.


```python
# Import the sklearn libarary
from sklearn.preprocessing import Imputer
# Instanciate the Imputer class
# misstion_value: the place holder for the missting value, here use the default 'NaN'
# strategy: 'mean', 'median' and 'most_frequent'
# aixs: 0 - impute along columns. 1 - impute along rows.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Fit column Age and Salary
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

```


```python
# value of X
print(X)
```

    [['France' 44.0 72000.0]
     ['Spain' 27.0 48000.0]
     ['Germany' 30.0 54000.0]
     ['Spain' 38.0 61000.0]
     ['Germany' 40.0 63777.77777777778]
     ['France' 35.0 58000.0]
     ['Spain' 38.77777777777778 52000.0]
     ['France' 48.0 79000.0]
     ['Germany' 50.0 83000.0]
     ['France' 37.0 67000.0]]


Here we used sklearn's Imputer module to help us to take care of the missing data. From the code you can see it become very easy by using the libaray. And the missing Age and Salary are filled with the mean value of their column.

## Encoding Categorical Data

In our dataset, the first column is contry name. The values in this column are text not numbers. But the machine learning model only work with numbers. So we need to encode the country name into numbers. 


```python
# Import LabelEncoder to encode text into numbers
from sklearn.preprocessing import LabelEncoder
# Encode first column of X
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

# print value of X
print(X)
```

    [[0 44.0 72000.0]
     [2 27.0 48000.0]
     [1 30.0 54000.0]
     [2 38.0 61000.0]
     [1 40.0 63777.77777777778]
     [0 35.0 58000.0]
     [2 38.77777777777778 52000.0]
     [0 48.0 79000.0]
     [1 50.0 83000.0]
     [0 37.0 67000.0]]


After using LabelEncoder, you can see we encode the country names from text into numbers. Here we got 'France' -> 0, 'Germany' -> 1, 'Spain' -> 2. All good, right? NO! Here's the problem, by encoding 'France', 'Germany' and 'Spain' into 0, 1 and 2, it means 'Spain' is greater than 'Germany', and 'Germany' is greater than 'France', just like 2 > 1 >0. This is wrong and all the countries should be considered on the same level. So we need to do some extra work to correct this result. 


```python
# Import OneHotEncoder module
from sklearn.preprocessing import OneHotEncoder
# Instantiate OneHotEncoder and set the first column 
oneHotEncoder = OneHotEncoder(categorical_features = [0])
# Encoder the first column of X and return an array object
X = oneHotEncoder.fit_transform(X).toarray()

# print value of X
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter = {'float_kind' : float_formatter})
print(X)
```

    [[1.00 0.00 0.00 44.00 72000.00]
     [0.00 0.00 1.00 27.00 48000.00]
     [0.00 1.00 0.00 30.00 54000.00]
     [0.00 0.00 1.00 38.00 61000.00]
     [0.00 1.00 0.00 40.00 63777.78]
     [1.00 0.00 0.00 35.00 58000.00]
     [0.00 0.00 1.00 38.78 52000.00]
     [1.00 0.00 0.00 48.00 79000.00]
     [0.00 1.00 0.00 50.00 83000.00]
     [1.00 0.00 0.00 37.00 67000.00]]


After the categorical encoding, the first column of X is separated into three columns. Each column represent one country. By doing this the model will know which country the customer comes from and make sure the model treat all the country at the same level. Now let's encode y as well.


```python
# before encoding
print('Before encoding: ')
print(y)

# Since y only has two values, 'yes' and 'no', we can just simply encode the value to 1 and 0 
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# print y
print('After encoding: ')
print(y)
```

    Before encoding: 
    ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
    After encoding: 
    [0 1 0 0 1 1 0 1 0 1]


## Splitting the data set into the Training set and Test set

One thing that every machine learning process will do is to split dataset into Training set and Test set. Just like human, if the machine keep learning the same dataset, it could "learning it by heart". Which means that the model could perform very accurate prediction on the same dataset, but when given a new dataset it model just performs poorly. We call this kind of scenario "overfitting". As a result, to avoid this situation we'd like to separate the dataset into Training set which will be used for training the model. And Test set, which test if the model's performance. If the model performs poorly on the Test set then we can try to correct the settings in the model and retry it. In some cases, people even break the dataset into three portions, training set, test set and validation set. So after trained and tested, validation set can be used to verify the final performance of the model. And this method to validate the model called cross validation.  


```python
# Import 'train_test_split', a very self explaining module
# Please notice the libaray we use is 'model_selection' from 'sklearn'
from sklearn.model_selection import train_test_split
# Split X into X_train and X_test. Split y into y_train and y_test
# test_size: usually choose less than 0.5 of the full dataset
# random_state: when splitting the dataset we want make the dataset to a random order first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Print out values
print('X_train: ')
print(X_train)
print()
print('X_test: ')
print(X_test)
print()
print('y_train: ')
print(y_train)
print()
print('y_test: ')
print(y_test)
```

    X_train: 
    [[0.00 1.00 0.00 40.00 63777.78]
     [1.00 0.00 0.00 37.00 67000.00]
     [0.00 0.00 1.00 27.00 48000.00]
     [0.00 0.00 1.00 38.78 52000.00]
     [1.00 0.00 0.00 48.00 79000.00]
     [0.00 0.00 1.00 38.00 61000.00]
     [1.00 0.00 0.00 44.00 72000.00]
     [1.00 0.00 0.00 35.00 58000.00]]
    
    X_test: 
    [[0.00 1.00 0.00 30.00 54000.00]
     [0.00 1.00 0.00 50.00 83000.00]]
    
    y_train: 
    [1 1 1 0 1 0 0 1]
    
    y_test: 
    [0 0]


## Feature Scaling 

Feature scaling is another must do step for most of the data preprocessing. What it dose is to scale the values in a dataset into a range of -1 ~ 1. To do this has two benefits, firt the computation time is shorter with small scale numbers. Second, is to avoid a certain feature dominate the result due to a bigger scale. For example, in our dataset there are Age and Salary features. The values of Salary are much bigger than Age, so during the training process Salary feature has a chance to dominate the result and make Age feature become useless. So by scale them into the same level, we can avoid this problem to happen.

There are two main ways to scale the feature: Standardization and Normalization.


Standardization: $$x' = \frac{x - mean(X)}{standard_-deviation_-(x)}$$

Normalization : $$x' = \frac{x - mean(x)}{max(x) - min(x)}$$


```python
# Here I use Standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 3:5] = sc_X.fit_transform(X_train[:, 3:5])
X_test[:, 3:5] = sc_X.transform(X_test[:, 3:5])

# Print out X_train, X_test
print('X_train: ')
print(X_train)
print()
print('X_test: ')
print(X_test)
```

    X_train: 
    [[0.00 1.00 0.00 0.26 0.12]
     [1.00 0.00 0.00 -0.25 0.46]
     [0.00 0.00 1.00 -1.98 -1.53]
     [0.00 0.00 1.00 0.05 -1.11]
     [1.00 0.00 0.00 1.64 1.72]
     [0.00 0.00 1.00 -0.08 -0.17]
     [1.00 0.00 0.00 0.95 0.99]
     [1.00 0.00 0.00 -0.60 -0.48]]
    
    X_test: 
    [[0.00 1.00 0.00 -1.46 -0.90]
     [0.00 1.00 0.00 1.98 2.14]]


Now we have all the step implemented for data preprocessing. In practice, not all the steps are needed. Please select the required steps based on your dataset. Below is the full version of the code.

``` python
# Data Preprocessing

# Import the libararies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```
