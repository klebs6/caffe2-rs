crate::ix!();

use crate::{
    OperatorStorage,
    GradientMakerBase,
    OperatorDef,
    CUDAContext,
};

/**
  | Scale takes one input data (Tensor)
  | and produces one output data (Tensor)
  | whose value is the input data tensor
  | scaled element-wise.
  |
  */
pub struct ScaleOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    scale:  f32,
}

register_cpu_operator!{Scale, ScaleOp<CPUContext>}

num_inputs!{Scale, 1}

num_outputs!{Scale, 1}

args!{Scale, 
    0 => ("scale", "(float, default 1.0) the scale to apply.")
}

identical_type_and_shape!{Scale}

allow_inplace!{Scale, vec![(0, 0)]}

impl<Context> ScaleOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            scale_(this->template GetSingleArgument<float>("scale", 1.0))
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            auto& X = Input(0);

        auto* Y = Output(0, X.sizes(), at::dtype<T>());
        math::Scale<float, T, Context>(
            X.numel(),
            scale_,
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
}

pub struct GetScaleGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetScaleGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // CopyArguments is true by default so the "scale" arg is going to be copied
        return SingleGradientDef(
            "Scale", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{Scale, GetScaleGradient}

impl ScaleOp<CUDAContext> {

    #[inline] pub fn run_on_cuda_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<at::Half, float>>::call(this, Input(0));
        */
    }
}

register_cuda_operator!{Scale, ScaleOp<CUDAContext>}
