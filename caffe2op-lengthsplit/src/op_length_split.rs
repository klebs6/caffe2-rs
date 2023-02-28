crate::ix!();

use crate::{
    OperatorStorage,
};

/**
| Given input vector LENGTHS, and input n_split,
| LengthsSplit returns a single output vector. 
|
| It "splits" each length into n_split values which
| add up to the original length. 
|
| It will attempt to do equal splits, and if not
| possible, it orders larger values first. 
|
| If the n_split is larger than the length, zero
| padding will be applied.
|
| e.g. LENGTHS = [9 4 5]
|      n_split = 3
|      Y = [3 3 3 2 1 1 2 2 1]
|
| e.g. LENGTHS = [2, 1, 2]
|      n_split = 3
|      Y = [1 1 0 1 0 0 1 1 0]
*/
pub struct LengthsSplitOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    n_split: i32,
}

num_inputs!{LengthsSplit, (1,2)}

num_outputs!{LengthsSplit, 1}

inputs!{LengthsSplit, 
    0 => ("LENGTHS", "Mx1 Input tensor denoting INT32 lengths"),
    1 => ("n_split", "(Optional) Number of splits for each element in LENGTHS (overrides argument)")
}

outputs!{LengthsSplit, 
    0 => ("Y", "(M*n_split)x1 Output vector denoting split lengths")
}

args!{LengthsSplit, 
    0 => ("n_split", "Number of splits for each element in LENGTHS")
}

scalar_type!{LengthsSplit, TensorProto::INT32}

impl<Context> LengthsSplitOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            n_split_(OperatorStorage::GetSingleArgument<int32_t>("n_split", 0)) 

        if (InputSize() == 1) {
          // If not specified, then must have this argument
          CAFFE_ENFORCE(
              OperatorStorage::HasArgument("n_split"),
              "Argument `n_split` is missing and was not specified as input.");
          CAFFE_ENFORCE(
              n_split_ > 0,
              "`n_split` must contain a positive value for defined behavior.");
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& L = Input(0);
        CAFFE_ENFORCE_EQ(L.dim(), 1, "Input `LENGTHS` should be a 1D vector.");

        if (InputSize() > 1) {
          // We potentially have n_split specified as inputs as well
          CAFFE_ENFORCE(
              Input(1).dim() == 1 && Input(1).numel() == 1,
              "Input `n_split` should be a vector of size 1.");

          const auto& input1 = Input(1);
          context_.template CopyItems<Context, CPUContext>(
              input1.dtype(), 1, input1.raw_data(), &n_split_);
        }

        CAFFE_ENFORCE(
            n_split_ > 0,
            "`n_split` must contain a positive value for defined behavior.");
        const auto M = L.numel();

        auto* Y = Output(0, {M * n_split_}, at::dtype<int32_t>());

        const int32_t* Ldata = L.template data<int32_t>();
        int32_t* Ydata = Y->template mutable_data<int32_t>();

        for (int i = 0; i < M; i++) {
          int32_t mod = Ldata[i] % n_split_;
          int32_t res =
              mod != 0 ? math::DivUp(Ldata[i], n_split_) : Ldata[i] / n_split_ + 1;
          for (int j = 0; j < n_split_; j++) {
            Ydata[(i * n_split_) + j] = mod-- > 0 ? res : res - 1;
          }
        }
        return true;
        */
    }
}

register_cpu_operator!{LengthsSplit, LengthsSplitOp<CPUContext>}

// TODO: Write gradient for this when needed
gradient_not_implemented_yet!{LengthsSplit}
