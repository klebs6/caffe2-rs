crate::ix!();

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **sum**.
  | 
  | Can reduce more than one of the "first"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the sum operation.
  | 
  | - If input tensor `X` has shape $(d_0,
  | d_1, d_2, ..., d_n)$, `lengths` must
  | have shape $(d_1 * d_2 * ... * d_{n})$.
  | 
  | - The values of the `lengths` tensor
  | determine how many of the values to consider
  | for each vector in the $d_{0}$ dimension.
  | 
  | For example, if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$
  | and $lengths = [2,3,1,2]$, then $Y =
  | [sum(1,4), sum(5,1,7), sum(2), sum(9,2)]
  | = [2.5, 4.333, 2, 5.5]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_sum_ops.cc
  |
  */
register_cpu_operator!{ReduceFrontSum, SumReduceDimsOp<CPUContext, true, false>}

num_inputs!{ReduceFrontSum, (1,2)}

num_outputs!{ReduceFrontSum, 1}

inputs!{ReduceFrontSum, 
    0 => ("X",       "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceFrontSum, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceFrontSum, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceFrontSum, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
           REDUCTION_OP_SHAPE_INFERENCE(true)
        */
    }
}

///--------------------------------

register_cpu_operator!{ReduceFrontSumGradient,
    SumReduceDimsGradientOp<CPUContext, true, false>
}

num_inputs!{ReduceFrontSumGradient, (2,3)}

num_outputs!{ReduceFrontSumGradient, 1}

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **sum**.
  | 
  | Can reduce more than one of the "last"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the sum operation.
  | 
  | - If input tensor `X` has shape $(d_0,
  | d_1, d_2, ..., d_n)$, `lengths` must
  | have shape $(d_0 * d_1 * d_2 * ... * d_{n-1})$.
  | 
  | - The values of the `lengths` tensor
  | determine how many of the values to consider
  | for each vector in the $d_{n-1}$ dimension.
  | 
  | For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$
  | and $lengths = [2,3,1]$, then $Y = [sum(1,5),
  | sum(4,1,8), sum(2)] = [6, 13, 2]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_sum_ops.cc
  |
  */
register_cpu_operator!{ReduceBackSum,         SumReduceDimsOp<CPUContext, false, false>}

num_inputs!{ReduceBackSum, (1,2)}

num_outputs!{ReduceBackSum, 1}

inputs!{ReduceBackSum, 
    0 => ("X", "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceBackSum, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceBackSum, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceBackSum, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
           REDUCTION_OP_SHAPE_INFERENCE(false)
        */
    }
}

///------------------------
register_cpu_operator!{ReduceBackSumGradient, SumReduceDimsGradientOp<CPUContext, false, false>}

num_inputs!{ReduceBackSumGradient, (2,3)}

num_outputs!{ReduceBackSumGradient, 1}

/**
  | Computes the **sum** of the input tensor's
  | elements along the provided `axes`.
  | The resulting tensor has the same rank
  | as the input if the `keepdims` argument
  | equals 1 (default). If `keepdims` is
  | set to 0, then the `axes` dimensions
  | are pruned.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc
  |
  */
register_cpu_operator!{ReduceSum,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        SumReducer<CPUContext>>
}

num_inputs!{ReduceSum, 1}

num_outputs!{ReduceSum, 1}

inputs!{ReduceSum, 
    0 => ("X", "(*Tensor`<float>`*): input tensor")
}

outputs!{ReduceSum, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceSum, 
    0 => ("axes", "(*Tuple(int)*): list of axes to reduce"),
    1 => ("keepdims", "(*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)")
}

tensor_inference_function!{ReduceSum, ReduceShapeInference}

///-----------------------------------
register_cpu_operator!{ReduceSumGradient,
    ReduceGradientOp<
        TensorTypes<i32, i64, f32, f64>,
        CPUContext,
        SumReducer<CPUContext>>
}

num_inputs!{ReduceSumGradient, 3}

num_outputs!{ReduceSumGradient, 1}
