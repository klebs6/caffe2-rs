crate::ix!();

use crate::{
    TanhGradientFunctor,
    GradientMakerBase,
    OperatorDef,
    CPUContext,
};

impl TanhGradientFunctor<CPUContext> {

    #[inline] pub fn forwardf32(&self, 
        y_dims:  &Vec<i32>,
        dy_dims: &Vec<i32>,
        y:       *const f32,
        dy:      *const f32,
        dx:      *mut f32,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            const int size = std::accumulate(
          Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
      ConstEigenVectorArrayMap<float> dY_arr(dY, size);
      ConstEigenVectorArrayMap<float> Y_arr(Y, size);
      EigenVectorMap<float>(dX, size) = dY_arr * (1 - Y_arr * Y_arr);
      return true;
        */
    }
}

register_cpu_operator!{
    TanhGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        TanhGradientFunctor<CPUContext>>}

pub struct GetTanhGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetTanhGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "TanhGradient",
            "",
            std::vector<std::string>{O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Tanh, GetTanhGradient}

register_cudnn_operator!{Tanh,         CudnnActivationOp<CUDNN_ACTIVATION_TANH>}

register_cudnn_operator!{TanhGradient, CudnnActivationGradientOp<CUDNN_ACTIVATION_TANH>}
