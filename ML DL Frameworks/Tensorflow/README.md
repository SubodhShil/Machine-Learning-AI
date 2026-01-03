# TensorFlow Learning Journey

<details>
<summary>Day 1: Introduction to TensorFlow</summary>

### Getting Started with TensorFlow

TensorFlow is a powerful open-source library developed by Google for numerical computation and large-scale machine learning. It allows you to build and train machine learning models, especially deep learning models.

First, you'll need to make sure you have TensorFlow installed. You can install it using pip:

```sh
pip install tensorflow
```

Now, let's write your first bit of TensorFlow code. This will verify that TensorFlow is installed and running correctly.

Here is a simple "Hello, World" example. You can add this code to a cell in your `01.ipynb` file and run it.

```python
import tensorflow as tf

# Print the TensorFlow version
print("TensorFlow version:", tf.__version__)

# Create a constant tensor
hello = tf.constant("Hello, TensorFlow!")

# Print the tensor
print(hello)

# To see the actual string value, you can use .numpy()
print(hello.numpy().decode('utf-8'))
```

### Explanation:

1.  `import tensorflow as tf`: This line imports the TensorFlow library and gives it the alias `tf`, which is a standard convention.
2.  `tf.__version__`: This prints the version of TensorFlow you have installed.
3.  `tf.constant("...")`: This creates a TensorFlow **tensor**. A tensor is the fundamental data structure in TensorFlow, similar to a multi-dimensional array. In this case, it's a constant tensor holding a string.
4.  `print(hello)`: When you print the tensor object, you'll see some information about it, including its value, shape, and data type.
5.  `hello.numpy()`: To get the actual value out of the tensor (in this case, the string), you can call the `.numpy()` method on it. This returns a NumPy representation of the tensor's value. We use `.decode('utf-8')` to see the string nicely formatted.

</details>

<details>
<summary>Day 2: Basic Tensor Operations</summary>

### Tensors

Tensors are the core data structure in TensorFlow. They are multi-dimensional arrays, similar to NumPy's ndarrays.

#### Creating Tensors

You can create tensors in various ways:

*   **Scalar (0-D tensor):** A single number.
*   **Vector (1-D tensor):** A list of numbers.
*   **Matrix (2-D tensor):** A table of numbers.
*   And so on for higher dimensions.

Here's how you can create them:

```python
# Scalar
scalar = tf.constant(7)
print(scalar)

# Vector
vector = tf.constant([10, 10])
print(vector)

# Matrix
matrix = tf.constant([[1, 2], [3, 4]])
print(matrix)
```

### Basic Arithmetic

You can perform element-wise arithmetic operations on tensors.

```python
# Create a tensor
tensor = tf.constant([[1, 2], [3, 4]])

# Add 10 to each element
print(tensor + 10)

# Multiply each element by 2
print(tensor * 2)

# Element-wise multiplication
print(tensor * tensor)
```

### Matrix Multiplication

For matrix multiplication, you can use `tf.matmul()`.

```python
matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[5, 6], [7, 8]])

# Matrix multiplication
product = tf.matmul(matrix1, matrix2)
print(product)
```

</details>

<details>
<summary>Day 3: TensorFlow Variables</summary>

### Introduction to Variables

While `tf.constant` creates immutable tensors, `tf.Variable` is used to create mutable tensors that can be changed during the execution of a program. This is essential for machine learning, as model parameters (like weights and biases) need to be updated during training.

#### Creating Variables

You can create a variable from any tensor-like object.

```python
# Create a variable
changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])

print(changeable_tensor)
print(unchangeable_tensor)
```

#### Changing Variables

The primary way to change the value of a `tf.Variable` is by using the `.assign()` method.

```python
# This will work
changeable_tensor[0].assign(7)
print(changeable_tensor)

# This will fail because constants are immutable
# unchangeable_tensor[0].assign(7) # This would cause an error
```

You can also use other assignment methods like `.assign_add()` and `.assign_sub()`.

```python
# Add 3 to the first element
changeable_tensor[0].assign_add(3)
print(changeable_tensor)

# Subtract 2 from the second element
changeable_tensor[1].assign_sub(2)
print(changeable_tensor)
```

Variables are crucial when you start building models that need to learn and update their parameters.

</details>

<details>
<summary>Day 4: More on Tensors (Shape and Random Values)</summary>

### Creating Tensors with Specific Shapes

Sometimes you need to create tensors with a specific shape, filled with ones or zeros.

```python
# Create a tensor of all ones
ones = tf.ones([2, 3])
print(ones)

# Create a tensor of all zeros
zeros = tf.zeros([3, 2])
print(zeros)
```

### Creating Random Tensors

Creating tensors with random values is very useful for initializing the weights of a neural network.

You can create random tensors from different distributions.

```python
# Create a tensor with random values from a normal distribution
random_normal = tf.random.normal([3, 3])
print(random_normal)

# Create a tensor with random values from a uniform distribution
random_uniform = tf.random.uniform([3, 3])
print(random_uniform)
```

</details>

<details>
<summary>Day 5: Tensor Attributes</summary>

### Getting Information from Tensors

It's often necessary to get information about a tensor's properties. TensorFlow provides several attributes to do this.

Let's create a tensor to work with:
```python
# Create a rank-4 tensor (4 dimensions)
rank_4_tensor = tf.zeros(shape=[2, 3, 4, 5])
```

#### Shape, Rank, and Size

*   **Shape**: The length (number of elements) of each of the dimensions of a tensor.
*   **Rank**: The number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix is rank 2, and so on.
*   **Axis or Dimension**: A particular dimension of a tensor.
*   **Size**: The total number of items in the tensor.

```python
print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimensions (rank):", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along the 0 axis:", rank_4_tensor.shape[0])
print("Elements along the last axis:", rank_4_tensor.shape[-1])
print("Total number of elements:", tf.size(rank_4_tensor).numpy())
```

### Indexing and Slicing Tensors

You can access specific elements of a tensor using indexing, much like Python lists or NumPy arrays.

```python
# Get the first 2 elements of each dimension
rank_4_tensor[:2, :2, :2, :2]

# Get the first element from each dimension from the end except for the last one
rank_4_tensor[:1, :1, :1, :]
```

These attributes and operations are fundamental for debugging and building models, as you often need to ensure that the dimensions of your tensors are what you expect.

</details>

<details>
<summary>Day 6: Manipulating Tensors</summary>

### Reshaping Tensors

One of the most common operations is changing the shape of your tensors. The `tf.reshape()` function is perfect for this, as long as the total number of elements in the tensor remains the same.

```python
# Create a tensor
tensor = tf.constant([[10, 7], [3, 4]]) # Shape (2, 2)

# Reshape to (4, 1)
reshaped_tensor = tf.reshape(tensor, shape=[4, 1])
print(reshaped_tensor)
```

### Expanding Dimensions

Sometimes, you need to add an extra dimension to your tensor. This is common when preparing data for a model that expects a certain number of dimensions (e.g., adding a batch dimension or a channel dimension). `tf.expand_dims()` is used for this.

```python
# Create a tensor
tensor = tf.constant([1, 2, 3]) # Shape (3,)

# Expand dimensions at axis 0
expanded_tensor = tf.expand_dims(tensor, axis=0)
print(expanded_tensor) # Shape is now (1, 3)

# Expand dimensions at axis 1
expanded_tensor_axis1 = tf.expand_dims(tensor, axis=1)
print(expanded_tensor_axis1) # Shape is now (3, 1)
```

### Squeezing Tensors

The opposite of expanding is squeezing, where you remove dimensions of size 1. `tf.squeeze()` handles this.

```python
# Create a tensor with a dimension of 1
squeezable_tensor = tf.constant([[[1], [2], [3]]]) # Shape (1, 3, 1)

# Squeeze the tensor
squeezed_tensor = tf.squeeze(squeezable_tensor)
print(squeezed_tensor) # Shape is now (3,)
```

These manipulation functions are essential for getting your data into the right shape for your models.

</details>

<details>
<summary>Day 7: One-Hot Encoding</summary>

### What is One-Hot Encoding?

One-hot encoding is a process of converting categorical data variables so they can be provided to machine learning algorithms to improve predictions.

For example, if you have a feature "color" with values "red", "green", and "blue", you can't just use these strings in a model. You would first convert them to integers, say `red=0`, `green=1`, `blue=2`.

However, this integer representation can be misleading. A model might assume that "green" (1) is somehow halfway between "red" (0) and "blue" (2). To avoid this, we use one-hot encoding, which transforms the single integer column into multiple columns, where only one is "hot" (1) at a time.

*   `red`: `[1, 0, 0]`
*   `green`: `[0, 1, 0]`
*   `blue`: `[0, 0, 1]`

### One-Hot Encoding in TensorFlow

TensorFlow provides a simple way to perform one-hot encoding with `tf.one_hot()`.

```python
# Create a list of indices
some_list = [0, 1, 2, 3] # could be red, green, blue, purple

# One-hot encode our list of indices
tf.one_hot(some_list, depth=4)
```
The `depth` parameter specifies how many classes or categories there are.

You can also set the "on" and "off" values.
```python
# Specify custom values for on and off
tf.one_hot(some_list, depth=4, on_value="I love deep learning", off_value="I also like to dance")
```

This is a fundamental technique for preparing categorical features for your models.

</details>

<details>
<summary>Day 8: More Math Operations</summary>

### Finding the min, max, mean, sum (aggregation)

You can perform many common mathematical operations on your tensors.

```python
# Create a new tensor
E = tf.constant(np.random.randint(0, 100, size=50))
E
```

#### Get the minimum
```python
tf.reduce_min(E)
```

#### Get the maximum
```python
tf.reduce_max(E)
```

#### Get the mean
```python
tf.reduce_mean(E)
```

#### Get the sum
```python
tf.reduce_sum(E)
```

</details>

<details>
<summary>Day 9: Finding Positional Minimum and Maximum</summary>

### Finding the Index of the Max and Min Value

Sometimes you need to know the *position* or *index* of the maximum or minimum value in a tensor. This is extremely common in classification tasks, where the highest value in a tensor corresponds to the model's prediction.

`tf.argmax()` and `tf.argmin()` are the functions for this.

```python
# Create a new tensor
F = tf.constant(np.random.random(50))
F
```

#### Find the positional maximum
```python
# Find the index of the largest value
tf.argmax(F)
```

#### Find the positional minimum
```python
# Find the index of the smallest value
tf.argmin(F)
```

Think of a scenario where a model outputs probabilities for 10 different classes. The output might be a tensor of shape `(10,)`. To find out which class the model predicts, you would use `tf.argmax()` to find the index of the highest probability.

</details>

<details>
<summary>Day 10: Your First TensorFlow Model</summary>

### Introduction to `tf.keras`

`tf.keras` is the recommended high-level API for TensorFlow. It provides a simple and modular way to create and train neural networks.

### Building a Simple Model

The most common type of model is a stack of layers, which you can create using `tf.keras.Sequential`.

Let's start by building a very simple model for a regression task. Imagine we want to predict a single number (like a house price) based on a single input feature (like the size of the house).

First, let's create some sample data:
```python
import numpy as np

# Features (e.g., house size)
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Labels (e.g., house price)
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
```
As you can see, the relationship is `y = X + 10`. Let's see if a model can learn this.

Now, let's build the model:
```python
# Create a simple model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])
```
This is a very basic neural network. Let's break it down:
*   `tf.keras.Sequential`: This groups a linear stack of layers into a `tf.keras.Model`.
*   `tf.keras.layers.Dense(1)`: This is a **fully connected** neural network layer. The `1` means it has one neuron, which will output a single number.

### Compiling the Model

Before you can train a model, you need to configure it. This is done with the `.compile()` method, where you specify:
1.  **Loss function:** How the model's error is measured. For regression, `mae` (mean absolute error) is a good choice.
2.  **Optimizer:** How the model updates its internal patterns (weights) to reduce the loss. `sgd` (stochastic gradient descent) is a basic but effective optimizer.

```python
# Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])
```

### Training the Model

Now it's time to train the model, which means feeding it our data so it can learn the relationship between `X` and `y`. We do this with the `.fit()` method.

The `epochs` parameter tells the model how many times to go through the training data.

```python
# Train the model
model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)
```

And that's it! You've built, compiled, and trained your first neural network. The next step is to evaluate its performance and use it to make predictions.

</details>
