<ceil mathematical operator for DSP and machine
learning>

**Note: This crate is currently being translated from C++ to Rust, and some function bodies may still be in the process of translation.**

Ceiling (or "rounding up") is a mathematical
function that maps a real number to the smallest
integer greater than or equal to it. The ceil
operation is used in various mathematical
algorithms and is particularly useful in digital
signal processing and machine learning
computations where it can help discretize
continuous values to integer values.

This crate provides the CeilOp operator that can
be used in DSP and machine learning
computations. The operator takes a tensor as input
and applies the ceil function element-wise to each
value of the tensor. The resulting tensor has the
same shape as the input tensor with each value
rounded up to the nearest integer. The operator
can be used as a building block for various
algorithms that require discrete values or for
preprocessing continuous data in machine learning
applications.

This crate also provides support for feeding and
fetching tensors to and from a computation graph,
as well as linkingoperators in a computation
graph. It includes the astype function for
converting tensors to a different data type.

Overall, this crate provides a convenient and
efficient way to implement the ceil operation in
DSP and machine learning applications.

8> thanks! could you please elaborate more as to
when we use the ceil function in the context of
deep neural networks? what are some examples?

Sure, here are a few examples of when the ceil
function might be used in the context of deep
neural networks:

1. Convolutional Neural Networks (CNNs): In CNNs,
convolutional layers are used to extract
features from images. The output feature map
size is calculated by dividing the input size
by the stride of the convolution. If the result
is not an integer, the ceil function is applied
to round up to the nearest integer. This
ensures that the output size is large enough to
preserve all the information from the input.

2. Pooling Layers: Pooling layers are used to
downsample feature maps while retaining
important information. The output size of
a pooling layer is calculated by dividing the
input size by the stride. Again, if the result
is not an integer, the ceil function is applied
to round up to the nearest integer.

3. Spatial Transformers: Spatial transformer
networks (STNs) are used to spatially transform
images in a way that improves their spatial
invariance properties. In STNs, the output size
of the transformed image is calculated by
multiplying the input size by a transformation
matrix. If the result is not an integer, the
ceil function is applied to round up to the
nearest integer to ensure that no information
is lost during the transformation.

In general, the ceil function is used in deep
learning to ensure that the output size of an
operation is large enough to preserve all the
information from the input. This is especially
important when dealing with images and other
high-dimensional data where even small rounding
errors can result in the loss of important
features.

9> in mathematics more generally, how is the ceil
funciton used? could you please provide me some
examples?

Sure, here are some examples of how the ceil
function is used in mathematics more generally:

1. Rounding up: One of the most common uses of the
ceil function is to round a given number up to
the nearest integer. For example, if we have
a number `x = 4.2`, the ceil function of x is 5,
since it is the smallest integer greater than
or equal to x.

2. Computing bounds: In some mathematical
applications, we may need to find upper bounds
on certain expressions or functions. In such
cases, the ceil function can be used to obtain
these bounds. 

For example, suppose we have
a sequence of positive real numbers (a_n), and
we want to find an upper bound for the sum 
`S_n = a_1 + a_2 + ... + a_n`. 

One way to do this is to use the inequality 
`S_n <= n * ceil(max(a_1, a_2, ..., a_n))`, 

which gives us an upper bound on `S_n` in terms of
the maximum value of the sequence.

3. Discretization: In some mathematical models,
we may need to discretize a continuous variable
or function. The ceil function can be used to
round up the value of the continuous variable
to the nearest integer, which can then be used
as a discrete value. For example, if we are
modeling the temperature of a room, we may want
to discretize it into intervals of 1 degree
Celsius, and use the ceil function to round up
the temperature to the nearest integer.

4. Random number generation: In some cases, the
ceil function can be used to generate random
integers. For example, suppose we want to
generate a random integer between 1 and 10. 

We can first generate a random real number
x between 0 and 1, and then compute `ceil(10 * x)`
to obtain a random integer between 1 and 10.

These are just a few examples of how the ceil
function can be used in mathematics. Its
versatility and usefulness make it
a fundamental tool in many mathematical
applications.
