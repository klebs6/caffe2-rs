crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    OperatorDef,
};

/**
Reshape the tensor by prepending a dimension of fixed size and dividing the
size of the next dimension by that amount.
*/
pub struct PrependDimOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    dim_size: i64,
}

num_inputs!{PrependDim, 1}

num_outputs!{PrependDim, 1}

inputs!{PrependDim, 
    0 => ("data", "An input tensor.")
}

outputs!{PrependDim, 
    0 => ("reshaped", "Reshaped tensor.")
}

args!{PrependDim, 
    0 => ("dim_size", "Size of the dimension to prepend.")
}

allow_inplace!{PrependDim, vec![(0, 0)]}

impl<Context> PrependDimOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            dim_size_(this->template GetSingleArgument<int64_t>("dim_size", 0)) 

        CAFFE_ENFORCE_GT(
            dim_size_, 0, "Argument dim_size must be greater than zero.");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);

        CAFFE_ENFORCE(input.dim() > 0, "Input must be at least 1D.");
        CAFFE_ENFORCE(
            input.size(0) % dim_size_ == 0,
            "First dimension must be multiple of prepend_dim. Current first dimension: ",
            input.size(0));

        vector<int64_t> actual_new_shape(input.dim() + 1);
        actual_new_shape[0] = dim_size_;
        actual_new_shape[1] = input.size(0) / dim_size_;
        for (int i = 1; i < input.sizes().size(); ++i) {
          actual_new_shape[i + 1] = input.size(i);
        }
        output->Resize(actual_new_shape);

        if (output != &input) {
          // If we are not doing in-place computation, a copy is needed.
          context_.CopyItemsSameDevice(
              input.dtype(),
              input.numel(),
              input.raw_data(),
              output->raw_mutable_data(input.dtype()));
        }
        return true;
        */
    }
}

///---------------------------------------------
/**
Merge first two dimensions in a single dimension with size dim(0) * dim(1).
*/
pub struct MergeDimOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    dim_size: i64,
}

num_inputs!{MergeDim, 1}

num_outputs!{MergeDim, 1}

inputs!{MergeDim, 
    0 => ("data", "An input tensor.")
}

outputs!{MergeDim, 
    0 => ("reshaped", "Reshaped tensor.")
}

allow_inplace!{MergeDim, vec![(0, 0)]}

inherit_onnx_schema!{MergeDim, "Reshape"}

impl<Context> MergeDimOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);

        CAFFE_ENFORCE(input.dim() > 1, "Input must be at least 2D.");

        vector<int64_t> actual_new_shape(input.dim() - 1);
        actual_new_shape[0] = input.size(0) * input.size(1);
        for (int i = 1; i < input.sizes().size() - 1; ++i) {
          actual_new_shape[i] = input.size(i + 1);
        }
        output->Resize(actual_new_shape);

        if (output != &input) {
          // If we are not doing in-place computation, a copy is needed.
          context_.CopyItemsSameDevice(
              input.dtype(),
              input.numel(),
              input.raw_data(),
              output->raw_mutable_data(input.dtype()));
        }
        return true;
        */
    }
}

register_cpu_operator!{PrependDim, PrependDimOp<CPUContext>}

register_cpu_operator!{MergeDim,   MergeDimOp<CPUContext>}

pub struct GetPrependDimGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetPrependDimGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "MergeDim", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

impl<'a> CopyArguments for GetPrependDimGradient<'a> {

    /// Arguments are no longer needed in backprop.
    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_gradient!{PrependDim,      GetPrependDimGradient}

register_cuda_operator!{PrependDim, PrependDimOp<CUDAContext>}

register_cuda_operator!{MergeDim,   MergeDimOp<CUDAContext>}
