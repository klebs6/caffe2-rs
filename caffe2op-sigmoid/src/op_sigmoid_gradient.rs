crate::ix!();

impl SigmoidGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(&self, 
        y_dims:  &Vec<i32>,
        dy_dims: &Vec<i32>,
        y:       *const T,
        dy:      *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
          const int size = std::accumulate(
          Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> Y_arr(Y, size);
          EigenVectorArrayMap<T>(dX, size) = dY_arr * Y_arr * (T(1) - Y_arr);
          return true;
        */
    }
}

register_cpu_operator!{
    SigmoidGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        SigmoidGradientFunctor<CPUContext>
    >
}

pub struct GetSigmoidGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSigmoidGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SigmoidGradient",
            "",
            std::vector<std::string>{O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Sigmoid, GetSigmoidGradient}

register_cudnn_operator!{Sigmoid,         CudnnActivationOp<CUDNN_ACTIVATION_SIGMOID>}

register_cudnn_operator!{SigmoidGradient, CudnnActivationGradientOp<CUDNN_ACTIVATION_SIGMOID>}
