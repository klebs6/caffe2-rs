crate::ix!();

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **mean**.
  | 
  | Can reduce more than one of the "first"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the mean operation.
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
  | [mean(1,4), mean(5,1,7), mean(2),
  | mean(9,2)] = [2.5, 4.333, 2, 5.5]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_mean_ops.cc
  |
  */
register_cpu_operator!{
    ReduceFrontMean, 
    SumReduceDimsOp<CPUContext, true, true>
}

num_inputs!{ReduceFrontMean, (1,2)}

num_outputs!{ReduceFrontMean, 1}

inputs!{ReduceFrontMean, 
    0 => ("X", "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceFrontMean, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceFrontMean, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceFrontMean, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
           REDUCTION_OP_SHAPE_INFERENCE(true)
        */

    }
}

inherit_onnx_schema!{ReduceFrontMean, "ReduceMean"}

register_cpu_operator!{ReduceFrontMeanGradient, SumReduceDimsGradientOp<CPUContext, true, true>}

num_inputs!{ReduceFrontMeanGradient, (2,3)}

num_outputs!{ReduceFrontMeanGradient, 1}


num_inputs!{ReduceBackMean, (1,2)}

num_outputs!{ReduceBackMean, 1}

inputs!{ReduceBackMean, 
    0 => ("X", "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceBackMean, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceBackMean, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceBackMean, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
           REDUCTION_OP_SHAPE_INFERENCE(false)
        */
    }
}

inherit_onnx_schema!{ReduceBackMean, "ReduceMean"}

register_cpu_operator!{
    ReduceBackMean,         
    SumReduceDimsOp<CPUContext, false, true>
}

register_cpu_operator!{
    ReduceBackMeanGradient, 
    SumReduceDimsGradientOp<CPUContext, false, true>
}

num_inputs!{ReduceBackMeanGradient, (2,3)}

num_outputs!{ReduceBackMeanGradient, 1}
