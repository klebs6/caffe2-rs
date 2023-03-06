crate::ix!();

/**
  | This operator translates a list of indices
  | into a list of hashed indices.
  | 
  | A seed can be fed as an argument to change
  | the behavior of the hash function.
  | 
  | If a modulo is specified, all the hashed
  | indices will be modulo the specified
  | number. All input and output indices
  | are enforced to be positive.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct IndexHashOp<Context> {
    storage: OperatorStorage,
    context: Context,
    seed:    i64,
    modulo:  i64,
}

num_inputs!{IndexHash, 1}

num_outputs!{IndexHash, 1}

inputs!{IndexHash, 
    0 => ("Indices",          "Input feature indices.")
}

outputs!{IndexHash, 
    0 => ("HashedIndices",    "Hashed feature indices.")
}

args!{IndexHash, 
    0 => ("seed",             "seed for the hash function"),
    1 => ("modulo",           "must be > 0, hashed ids will be modulo this number")
}

allow_one_to_one_inplace!{IndexHash}

tensor_inference_function!{IndexHash, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          std::vector<TensorShape> out(1);
          std::vector<int64_t> output_dims = GetDimsVector(in[0]);
          out[0] = CreateTensorShape(output_dims, in[0].data_type());
          return out;
        */
    }
}

input_tags!{
    IndexHashOp {
        Indices
    }
}

output_tags!{
    IndexHashOp {
        HashedIndices
    }
}

register_cpu_operator!{IndexHash, IndexHashOp<CPUContext>}

should_not_do_gradient!{IndexHash}
