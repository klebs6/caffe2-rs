crate::ix!();

/**
  | Compute row-wise max reduction of the
  | input tensor.
  | 
  | This op takes one input, $X$, of shape
  | $BxMxN$, where $B$ is the batch size,
  | $M$ is number of rows, and $N$ is number
  | of columns.
  | 
  | The output of this op, $Y$, is a matrix
  | of shape $BxM$, with one row for each
  | element of the batch, and the same number
  | of columns as the number of rows of the
  | input tensor.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc
  |
  */
register_cpu_operator!{RowwiseMax, MaxReductionOp<f32, CPUContext, true>}

num_inputs!{RowwiseMax, 1}

num_outputs!{RowwiseMax, 1}

inputs!{RowwiseMax, 
    0 => ("X", "A tensor of dimensions $B x M x N$ to compute rowwise-max. 
        Here, $B$ is batch size, and $M$ and $N$ are the number of rows and columns of 
        each element of the batch, respectively.")
}

outputs!{RowwiseMax, 
    0 => ("Y", "The output tensor of shape $B x M$, where each row 
        represents the row-wise maximums for that element of the input batch.")
}
