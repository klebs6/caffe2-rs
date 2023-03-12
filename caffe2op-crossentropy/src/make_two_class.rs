crate::ix!();

/**
  | Given a vector of probabilities, this
  | operator transforms this into a 2-column
  | matrix with complimentary probabilities
  | for binary classification. In explicit
  | terms, given the vector X, the output
  | Y is vstack(1 - X, X).
  | 
  | Hacky: turns a vector of probabilities
  | into a 2-column matrix with complimentary
  | probabilities for binary classification
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MakeTwoClassOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    /**
      | Input: X
      | 
      | Output: Y = vstack(1-X, X)
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{MakeTwoClass, 1}

num_outputs!{MakeTwoClass, 1}

inputs!{MakeTwoClass, 
    0 => ("X", "Input vector of probabilities")
}

outputs!{MakeTwoClass, 
    0 => ("Y", "2-column matrix with complimentary probabilities of X for binary classification")
}

tensor_inference_function!{MakeTwoClass, /*[](const OperatorDef& /* unused */,
    const vector<TensorShape>& in) {
    vector<TensorShape> out(1);
    out[0].add_dims(in[0].dims(0));
    out[0].add_dims(2);
    return out;
}*/
}

register_cpu_operator!{MakeTwoClass, MakeTwoClassOp<f32, CPUContext>}
