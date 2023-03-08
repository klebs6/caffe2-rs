crate::ix!();

register_cpu_operator!{
    ReduceBackMax, 
    MaxReduceDimsOp<float, CPUContext, false>
}

num_inputs!{ReduceBackMax, (1,2)}

num_outputs!{ReduceBackMax, 1}

inputs!{ReduceBackMax, 
    0 => ("X", "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceBackMax, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceBackMax, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceBackMax, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
           REDUCTION_OP_SHAPE_INFERENCE(false)
           */
    }
}

register_cpu_operator!{
    ReduceBackMaxGradient,  
    MaxReduceDimsGradientOp<float, CPUContext, false>
}

num_inputs!{ReduceBackMaxGradient, (3,4)}

num_outputs!{ReduceBackMaxGradient, 1}
