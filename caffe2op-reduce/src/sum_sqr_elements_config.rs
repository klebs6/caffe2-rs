crate::ix!();

num_inputs!{SumSqrElements, 1}

num_outputs!{SumSqrElements, 1}

inputs!{SumSqrElements, 
    0 => ("X", "Tensor to sum up")
}

outputs!{SumSqrElements, 
    0 => ("sum", "Scalar sum of squares")
}

args!{SumSqrElements, 
    0 => ("average", "whether to average or not")
}

scalar_type!{SumSqrElements, TensorProto::FLOAT}
