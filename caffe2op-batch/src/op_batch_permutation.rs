crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    OperatorDef,
    Workspace,
    CPUContext
};

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
pub struct BatchPermutationOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
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

///--------------------------------------------------------------
pub struct BatchPermutationGradientOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

/**
  | Input: indices, dY (aka "gradOutput");
  | Output: dX (aka "gradInput")
  |
  */
num_inputs!{BatchPermutationGradient, 2}

num_outputs!{BatchPermutationGradient, 1}

register_cpu_operator!{
    BatchPermutationGradient, 
    BatchPermutationGradientOp<f32, CPUContext>
}

impl<T,Context> BatchPermutationGradientOp<T,Context> {
    
    pub fn new(def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(def, ws)
        */
    }
}

#[inline] pub fn batch_permutation_loop<const forwards: bool>(
    n:        i32,
    k:        i32,
    src:      *const f32,
    indices:  *const i32,
    dst:      *mut f32) 
{
    todo!();
    /*
        long numBytes = K * sizeof(float);
      if (forwards) {
    #ifdef _OPENMP
    #if (_OPENMP >= 201307)
    #pragma omp parallel for simd
    #else
    #pragma omp parallel for
    #endif
    #endif
        for (int n = 0; n < N; n++) {
          int origIdx = n * K;
          int permuteIdx = indices[n] * K;
          std::memcpy(dst + origIdx, src + permuteIdx, numBytes);
        }
      } else {
        std::vector<int> backward_indices(N);
        for (int i = 0; i < N; ++i) {
          backward_indices[indices[i]] = i;
        }
        for (int n = 0; n < N; n++) {
          int permuteIdx = n * K;
          int origIdx = backward_indices[n] * K;
          std::memcpy(dst + permuteIdx, src + origIdx, numBytes);
        }
      }
    */
}

impl BatchPermutationOp<f32, CPUContext> {

    #[inline] pub fn run_on_deviceA(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
          auto& indices = Input(1);

          CAFFE_ENFORCE(indices.dim() == 1, "indices must be 1-d");
          CAFFE_ENFORCE(
              X.dim32(0) == indices.dim32(0),
              "X.dim32(0) must be equal to indices.dim32(0)",
              "(",
              X.dim32(0),
              " vs. ",
              indices.dim32(0),
              ")");

          auto* Y = Output(0, X.sizes(), at::dtype<float>());

          if (X.dim32(0) > 0) {
            batch_permutation_loop<true>(
                X.dim32(0),
                X.numel() / X.dim32(0),
                X.data<float>(),
                indices.data<int>(),
                Y->mutable_data<float>());
          }
          return true;
        */
    }

    #[inline] pub fn run_on_deviceB(&mut self) -> bool {
        
        todo!();
        /*
            auto& indices = Input(0);
          auto& dY = Input(1);

          auto* dX = Output(0, dY.sizes(), at::dtype<float>());

          if (dY.dim32(0) > 0) {
            batch_permutation_loop<false>(
                dY.dim32(0),
                dY.numel() / dY.dim32(0),
                dY.data<float>(),
                indices.data<int>(),
                dX->mutable_data<float>());
          }
          return true;
        */
    }
}

pub struct GetBatchPermutationGradient;

impl GetGradientDefs for GetBatchPermutationGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BatchPermutationGradient",
            "",
            vector<string>{I(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{BatchPermutation, GetBatchPermutationGradient}

pub type BatchPermutationOpFloatCPU = BatchPermutationOp<f32, CPUContext>;
