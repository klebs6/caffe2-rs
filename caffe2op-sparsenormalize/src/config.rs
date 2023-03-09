crate::ix!();

/**
  | Given a sparse matrix, apply max_norm
  | or constant_norm sparse regularization.
  |
  */
register_cpu_operator!{
    SparseNormalize, 
    SparseNormalizeOp<float, CPUContext>
}

num_inputs!{SparseNormalize, (2,3)}

num_outputs!{SparseNormalize, 1}

inputs!{SparseNormalize, 
    0 => ("param",   "Parameters to be normalized"),
    1 => ("indices", "Sparse indices"),
    2 => ("grad",    "Gradient computed (optional - not used, this argument is for backwards compatibility)")
}

outputs!{SparseNormalize, 
    0 => ("output_param", "Normalized parameters")
}

args!{SparseNormalize, 
    0 => ("use_max_norm", "A bool variable to control whether to use max norm or constant norm. When use_max_norm = false, constant norm is used so that all the embedding vectors are scaled to have a L2 norm equals to A (see blow argument norm=A). If use_max_norm = true, max norm is used so that embedding is scaled so that its l2 norm is no larger than A. If an embedding's norm is less than A originally, the embedding is left unchanged.The default is True."),
    1 => ("norm",         "L2 norm of the embedding. The default is 1.0.")
}

enforce_one_to_one_inplace!{SparseNormalize}

should_not_do_gradient!{SparseNormalize}

/**
  | Given a sparse matrix, apply max_norm
  | or constant_norm sparse regularization.
  |
  */
register_cpu_operator!{Float16SparseNormalize, SparseNormalizeOp<c10::Half, CPUContext>}

num_inputs!{Float16SparseNormalize, (2,3)}

num_outputs!{Float16SparseNormalize, 1}

inputs!{Float16SparseNormalize, 
    0 => ("param",    "Parameters to be normalized"),
    1 => ("indices",  "Sparse indices"),
    2 => ("grad",     "Gradient computed (optional - not used, this argument is for backwards compatibility)")
}

outputs!{Float16SparseNormalize, 
    0 => ("output_param", "Normalized parameters")
}

args!{Float16SparseNormalize, 
    0 => ("use_max_norm", "A bool variable to control whether to use max norm or constant norm. When use_max_norm = false, constant norm is used so that all the embedding vectors are scaled to have a L2 norm equals to A (see blow argument norm=A). If use_max_norm = true, max norm is used so that embedding is scaled so that its l2 norm is no larger than A. If an embedding's norm is less than A originally, the embedding is left unchanged. The default is True."),
    1 => ("norm",         "L2 norm of the embedding. The default is 1.0.")
}

enforce_one_to_one_inplace!{Float16SparseNormalize}

should_not_do_gradient!{Float16SparseNormalize}

input_tags!{
    SparseNormalizeOp {
        Param,
        Indices
    }
}

output_tags!{
    SparseNormalizeOp {
        OutputParam
    }
}
