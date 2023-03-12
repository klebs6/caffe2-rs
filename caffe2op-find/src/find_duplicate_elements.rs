crate::ix!();

/**
  | The *FindDuplicateElements* op takes
  | a single 1-D tensor *data* as input and
  | returns a single 1-D output tensor *indices*.
  | The output tensor contains the indices
  | of the duplicate elements of the input,
  | excluding the first occurrences. If
  | all elements of *data* are unique, *indices*
  | will be empty.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_DISPATCH_HELPER]
pub struct FindDuplicateElementsOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{FindDuplicateElements, 1}

num_outputs!{FindDuplicateElements, 1}

inputs!{FindDuplicateElements, 
    0 => ("data", "a 1-D tensor.")
}

outputs!{FindDuplicateElements, 
    0 => ("indices", "Indices of duplicate elements in data, excluding first occurrences.")
}

register_cpu_operator!{
    FindDuplicateElements,
    FindDuplicateElementsOp<CPUContext>
}


should_not_do_gradient!{FindDuplicateElements}
