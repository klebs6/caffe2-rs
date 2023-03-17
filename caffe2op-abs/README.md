The "caffe2op-abs" crate is a Rust library that
provides an operator for absolute value
calculation. The crate includes functions for
calculating the gradient of the absolute value
function, which is useful in deep learning
applications. The crate also includes functions
for fetching and feeding data to and from tensors,
and for resetting the workspace.

**Note: This crate is currently being translated from C++ to Rust, and some function bodies may still be in the process of translation.**

The mathematical idea behind the absolute value
function is straightforward: it returns the
magnitude of a number, without regard to its
sign. For example, the absolute value of 5 and the
absolute value of -5 are both 5. The formula for
the absolute value function is:

| x | = {
  x    if x >= 0
 -x    if x < 0
}

The "randn" function in this crate generates
random numbers from a normal distribution, which
can be useful in deep learning applications. The
"astype" function is used to cast data from one
data type to another.
