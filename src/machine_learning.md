# Machine Learning

## Table of Contents

- [Numpy](#numpy)
- [Pandas](#pandas)
- [Matplotlib](#matplotlib)

## Numpy

Python includes a module named numpy that can be used to store data in a matrix-like object.

Import statement:

```python
import numpy as np
```

### What Is Numpy Array?

- A multi-dimensional array (data type = ndarray) can be created from a multi-dimensional list using the NumPy module.
- A one-dimensional array is an array that has only one dimension and contains elements of the same type and size.
- A two-dimensional array is an array that has two dimensions and contains elements of the same type and size.
- An array has “axis/axes” to indicate its dimensions.
- The first axis (axis = 0) of a 2-D array shows the number of rows and the second axis (axis = 1) shows the number of columns.
- Indexing or slicing the array can be used to get or change its elements, similar to lists.

### List To Numpy

```python
list1 = [1, 2, 3, 4, 5]
arr1 = npm.array(list1)
```

This also works for multi-dimensional lists, but only if the list elements have the same type (a list with both integers and floats will be converted to all floats)

### NumPy Indexing and Slicing

- You can index NumPy arrays similar to lists
- You can also slice NumPy arrays like lists

For a 2-D array:

```python
sub_arr = arr[start_row:end_row, start_col:end_col]
```

To access a whole row or column you can use empty slicing:

- `arr[:,0]` for all the rows in the first column
- `arr[0,:]` for all the columns in the first row

Example:

```python
arr = np.array([1,2,3,4,5,6])

print(arr[3]) #Output: 4

test1 = arr[3:]
test1[0] = 99

print(arr) # Output: [99, 5, 6]
```

### Filtering

To select array elements that meet a certain criterion, you can apply a `conditional expression`.

Example:

```python
# Creating a NumPy array with elements -2, -1, 0, 1, and 2
arr = np.array([-2, -1, 0, 1, 2])

# Selecting and displaying only the elements in the array that are greater than 0
selected_elements = arr[arr > 0] # Output: array([1, 2])
```

We could also use `np.where()`.

Example:

```python
# Creating a NumPy array with elements -2, -1, 0, 1, and 2
arr = np.array([-2, -1, 0, 1, 2])

# Using np.where to find the indices where the elements are greater than 0
indices = np.where(arr > 0)

# Selecting and displaying the elements that satisfy the condition using the obtained indices
selected_elements = arr[indices]
```

## Pandas

Pandas is a Python module used to import, export, and manipulate data.

import statement:

```python
import pandas as pd
```

### Data Types

When data is imported using pandas, there are two different data types depending on the dimensions:

- 1-D data is stored in a Seriess
- 2-D data is stored in a DataFrame
- Each column in a DataFrame represents a Series
- The values in each Series (data frame columns) must be the same type.

### ‘loc’ - Label-Based Indexing

The `loc` method in Pandas lets you access DataFrame data by labels or boolean array-based indexing.
Likewise, the `iloc` method lets you access DataFrame data by integer positions, like indexing elements in a Python list.

Example:

```python
# Selecting two rows and all columns that have the index values 'ID1' and 'ID3'
df.loc[['ID1', 'ID3'], :]

# Selecting multiple rows and columns where age is greater than 30 and then selecting the 'Name' and 'Age' columns
df.loc[df['Age'] > 30, ['Name', 'Age']]
```

### Identifying Missing Data

The `isna()` and `isnull()` methods are used interchangeably to check for missing values within a DataFrame or Series.

```python
df.isna()

# or

df.isnull()
```

### Filling Missing Data

The `fillna()` function is a versatile tool for replacing missing or NaN (Not a Number) values within a DataFrame or Series. Available methods are `ffill` for forward filling (propagating the last valid value forward) and `bfill` for backward filling (propagating the next valid value backward).

```python
# Backward fill

df.bfill()

# Forward fill

df.ffill()
```

### GroupBy

Pandas groupby is a method that splits the dataframe into groups based on one or more columns, applies a function to each group, and combines the results into a new DataFrame.

Example:

```python
Grouped = df.groupby('Category')

Result = Grouped.agg({'Value': ['mean', 'sum', 'count', 'max', 'min']})
```

## Matplotlib

Matplotlib is a built-in module in Python used for plotting.

Import statement:

```python
import matplotlib.pyplot

# or

import matplotlib.pyplot as plt
```

### Line Plot

Plots a line graph. It is commonly used for visualizing `continuous data`, like time
series or continuous functions.

```python
matplotlib.pyplot.plot()

# or

plt.plot()
```

### Scatter Plot

Scatter plots are used to visualize the relationship between two numerical variables, allowing you to `identify patterns, trends, clusters, correlations, and outliers`.

```python
matplotlib.pyplot.scatter()

# or

plt.scatter()
```

### Bar Plot

Bar plots are used to compare the values of different categories, display frequencies or counts of categorical variables, and visualize the relationship between categorical and numerical variables. `Useful for comparing discrete data`.

```python
matplotlib.pyplot.bar()

# or

plt.bar()
```
