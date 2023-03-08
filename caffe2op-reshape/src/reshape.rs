crate::ix!();

/**
  | Takes a shape and data tensor and reshapes
  | it
  | 
  | Reshape the input tensor similar to
  | numpy's [reshape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html).
  | 
  | Takes a tensor as input and an optional
  | tensor specifying the new shape. When
  | the second input is absent, an extra
  | argument shape must be specified. Outputs
  | the reshaped tensor as well as the original
  | shape.
  | 
  | At most one dimension of the new shape
  | can be -1. In this case, the value is inferred
  | from the size of the tensor and the remaining
  | dimensions. A dimension could also
  | be 0, in which case the actual dimension
  | value is going to be copied from the input
  | tensor.
  | 
  | For empty tensor, we will set the -1 dimension
  | to be 0 (if one dimension is -1).
  | 
  | When the tensor is empty, dimension
  | of 0 will remain to be 0.
  | 
  | E.g: data=np.empty(shape=[4, 0]),
  | shape=[0, -1], the output tensor will
  | be np.emtpy(shape=[0, 0])
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reshape_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ReshapeOp<F,Context> {
    storage:   OperatorStorage,
    context:   Context,
    new_shape: Vec<i64>,
    phantomF:  PhantomData<F>,
}
