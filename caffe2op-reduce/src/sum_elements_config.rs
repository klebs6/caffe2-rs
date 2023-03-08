crate::ix!();

num_inputs!{SumElements, 1}

num_outputs!{SumElements, 1}

inputs!{SumElements, 
    0 => ("X", "(*Tensor`<float>`*): blob pointing to an instance of a counter")
}

outputs!{SumElements, 
    0 => ("sum", "(*Tensor`<float>`*): Scalar tensor containing the sum (or average)")
}

args!{SumElements, 
    0 => ("average", "(*bool*): set to True to compute the average of the elements rather than the sum")
}

scalar_type!{SumElements, TensorProto::FLOAT}

///--------------------------

num_inputs!{SumElementsInt, 1}

num_outputs!{SumElementsInt, 1}

inputs!{SumElementsInt, 
    0 => ("X", "Tensor to sum up")
}

outputs!{SumElementsInt, 
    0 => ("sum", "Scalar sum")
}

scalar_type!{SumElementsInt, TensorProto::INT32}

should_not_do_gradient!{SumElementsInt}
