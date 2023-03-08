crate::ix!();

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **max**.
  | 
  | Can reduce more than one of the "first"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the max operation.
  | 
  | - If input tensor `X` has shape $(d_0,
  | d_1, d_2, ..., d_n)$, `lengths` must
  | have shape $(d_1 * d_2 * ... * d_{n})$.
  | 
  | - The values of the `lengths` tensor
  | determine how many of the values to consider
  | for each vector in the $d_{0}$ dimension.
  | 
  | For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$
  | and $lengths = [2,3,1,2]$, then $Y =
  | [max(1,4), max(5,1,7), max(2), max(9,2)]
  | = [4, 7, 2, 9]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_max_ops.cc
  |
  */
register_cpu_operator!{ReduceFrontMax, MaxReduceDimsOp<float, CPUContext, true>}

num_inputs!{ReduceFrontMax, (1,2)}

num_outputs!{ReduceFrontMax, 1}

inputs!{ReduceFrontMax, 
    0 => ("X", "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceFrontMax, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceFrontMax, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceFrontMax, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
        REDUCTION_OP_SHAPE_INFERENCE(true)
        */
    }
}

register_cpu_operator!{ReduceFrontMaxGradient, MaxReduceDimsGradientOp<float, CPUContext, true>}

num_inputs!{ReduceFrontMaxGradient, (3,4)}

num_outputs!{ReduceFrontMaxGradient, 1}

