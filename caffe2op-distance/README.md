## Dot Product

The dot product (also known as the scalar product
or inner product) is a binary operation that takes
two vectors of equal dimension and returns
a scalar. For two vectors `a` and `b` with `n`
components, the dot product is defined as:

```
a · b = ∑_{i=1}^n a_i b_i
```

where `a_i` and `b_i` denote the `i`-th components
of the vectors `a` and `b`, respectively.

The dot product is often used in mathematics,
physics, and engineering for a variety of
applications. In physics, it is used to calculate
the work done by a force on an object, while in
mathematics it is used to find the angle between
two vectors and to calculate the norm of
a vector. In engineering, it is used for tasks
such as signal processing, computer graphics, and
machine learning.

For example, in machine learning, the dot product
is often used as a measure of similarity between
two vectors. When the dot product between two
vectors is high, it indicates that the vectors are
similar in some way, while a low dot product
indicates dissimilarity.

Overall, the dot product is a fundamental
mathematical operation with a wide range of
applications in various fields of study.

### **Gradient for Dot Product:**

Let `X` and `Y` be two input vectors of equal size
`n`. The gradient of the `dot product` operator
with respect to the inputs `X` and `Y` can be
computed as:

`d(X.Y)/dX = Y`

`d(X.Y)/dY = X`

where `.` denotes the dot product.

Intuitively, these equations show that when we
change one element of `X`, the output of the dot
product changes by the corresponding element of
`Y`, and vice versa.

## Cosine Similarity

The cosine similarity is a measure of similarity
between two non-zero vectors of an inner product
space. It measures the cosine of the angle between
them and determines whether the two vectors are
pointing in roughly the same direction. The cosine
similarity is defined as:

```
cosine_similarity(x, y) = (x . y) / (||x|| ||y||)
```

Where `x` and `y` are two vectors, `.` denotes the
dot product and `||.||` denotes the Euclidean
norm. The output of the cosine similarity function
is a value between -1 and 1, where a value of
1 indicates that the two vectors are identical,
0 indicates that the vectors are orthogonal, and
-1 indicates that the two vectors are
diametrically opposed.

Cosine similarity is commonly used as a similarity
measure in many fields, such as:

- Information retrieval: To measure the similarity
  between documents or search queries.

- Machine learning: To compare feature vectors,
  for example in text classification,
  recommendation systems, or image recognition.

- Signal processing: To compare signals or
  frequency spectra, for example in audio or
  speech recognition.

- Social network analysis: To measure similarity
  between users or content in social networks.

In general, cosine similarity is a useful measure
of similarity when the magnitude of the vectors is
not important, but only their direction.


### **Gradient for Cosine Similarity:**

Let `X` and `Y` be two input vectors of equal size
`n`. The gradient of the `cosine similarity`
operator with respect to the inputs `X` and `Y`
can be computed as:

`d(cosine_similarity(X,Y))/dX = (Y - cosine_similarity(X,Y)*X)/||X||_2^2`

`d(cosine_similarity(X,Y))/dY = (X - cosine_similarity(X,Y)*Y)/||Y||_2^2`

where 
`cosine_similarity(X,Y) = (X.Y)/(||X||_2 * ||Y||_2)` 
is the cosine similarity between `X` and `Y`, and 
`||X||_2` denotes the `L2-norm` of vector `X`.

Intuitively, these equations show that the
gradients depend not only on the dot product of
the inputs `X` and `Y`, but also on their
individual magnitudes. The gradients indicate how
much changing one element of `X` or `Y` affects
the output of the `cosine similarity` operator,
and take into account the overall magnitude of
each input vector.


## L1 Distance

The L1 distance, also known as Manhattan distance
or taxicab distance, between two vectors `u` and
`v` of length `n` is defined as:

```
L1(u, v) = ||u - v||_1 = ∑_{i=1}^n |u_i - v_i|
```

The gradient of the L1 distance with respect to
`u` can be computed as:

```
∂L1(u, v) / ∂u_i = sign(u_i - v_i)
```

where `sign(x)` returns `-1` if `x < 0`, `0` if `x = 0`, 
and `1` if `x > 0`. 

The gradient with respect to `v` is the negative
of the gradient with respect to `u`, i.e.,

```
∂L1(u, v) / ∂v_i = - sign(u_i - v_i)
```

## Squared L2 Distance

The squared L2 distance between two vectors `u`
and `v` of length `n` is defined as:

```
squared_L2(u, v) = ||u - v||_2^2 = ∑_{i=1}^n (u_i - v_i)^2
```

The gradient of the squared L2 distance with
respect to `u` can be computed as:

```
∂squared_L2(u, v) / ∂u_i = 2(u_i - v_i)
```

The gradient with respect to `v` is the negative
of the gradient with respect to `u`, i.e.,

```
∂squared_L2(u, v) / ∂v_i = -2(u_i - v_i)
```

