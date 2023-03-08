crate::ix!();

/**
  | Compute column-wise max reduction
  | of the input tensor.
  | 
  | This op takes one input, $X$, of shape
  | $BxMxN$, where $B$ is the batch size,
  | $M$ is number of rows, and $N$ is number
  | of columns.
  | 
  | The output of this op, $Y$, is a matrix
  | of shape $BxN$, with one row for each
  | element of the batch, and the same number
  | of columns as the input tensor.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc
  |
  */
register_cpu_operator!{ColwiseMax, MaxReductionOp<f32, CPUContext, false>}

num_inputs!{ColwiseMax, 1}

num_outputs!{ColwiseMax, 1}

inputs!{ColwiseMax, 
    0 => ("X", "A tensor of dimensions $B x M x N$ to compute columnwise-max. 
        Here, $B$ is batch size, and $M$ and $N$ are the number of rows and columns of 
        each element of the batch, respectively.")
}

outputs!{ColwiseMax, 
    0 => ("Y", "The output tensor of shape $B x N$, where each row represents the 
        column-wise maximums for that element of the input batch.")
}

tensor_inference_function!{ColwiseMax, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
      vector<int64_t> output_dims = {in[0].dims()[0], in[0].dims()[2]};
      return vector<TensorShape>{
          CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};

        */
    }
}
