crate::ix!();

#[test] fn batch_permutation_op_example1() {

    /*
    Example of batch permutation on a 2-D tensor with batch size 4:
      X = [
        [1, 5, 2, 3, 4, 6, 0],
        [4, 3, 3, 5, 2, 3, 1],
        [2, 2, 3, 6, 0, 0, 1],
        [0, 0, 1, 1, 2, 2, 3]
      ]
      indices = [2, 0, 1, 3]
      Y = [
        [2, 2, 3, 6, 0, 0, 1],
        [1, 5, 2, 3, 4, 6, 0],
        [4, 3, 3, 5, 2, 3, 1],
        [0, 0, 1, 1, 2, 2, 3]
      ]
    */
}

#[test] fn batch_permutation_op_example2() {

    todo!();
    /*
    Example of batch permutation on a 3-D tensor with batch size 4:
      X = [
        [[1, 5, 2], [3, 4, 6, 0]],
        [[4, 3, 3], [5, 2, 3, 1]],
        [[2, 2, 3], [6, 0, 0, 1]],
        [[0, 0, 1], [1, 2, 2, 3]]
      ]
      indices = [2, 0, 1, 3]
      Y = [
        [[2, 2, 3], [6, 0, 0, 1]],
        [[1, 5, 2], [3, 4, 6, 0]],
        [[4, 3, 3], [5, 2, 3, 1]],
        [[0, 0, 1], [1, 2, 2, 3]]
      ]
    */
}

/**
  | Batch permutation of an input tensor
  | X given input indices.
  | 
  | First dimension of X equals batch size
  | N.
  | 
  | The indices stores a be permutation
  | of N.
  | 
  | The output Y is a tensor of same shape
  | as X, with data re-ordered according
  | to the indices within the batch size.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BatchPermutationOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

// Input: X, indices; Output: Y
num_inputs!{BatchPermutation, 2}

num_outputs!{BatchPermutation, 1}

inputs!{BatchPermutation, 
    0 => ("X", "Input tensor, where 1st dimension equals batch size"),
    1 => ("indices", "Input indices of batch to permute")
}

outputs!{BatchPermutation, 
    0 => ("Y", "Output permuted tensor")
}

#[cfg(caffe2_use_mkldnn)]
register_ideep_operator!{
    BatchPermutation, 
    IDEEPFallbackOp<BatchPermutationOp<f32, CPUContext>>
}

register_cpu_operator!{
    BatchPermutation, 
    BatchPermutationOp<f32, CPUContext>
}

impl<T,Context> BatchPermutationOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}
