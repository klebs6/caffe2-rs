crate::ix!();

/**
  | Calculates the tangent of the given
  | input tensor, element-wise.
  |
  */
pub struct TanFunctor<Context> { 

    phantom: PhantomData<Context>,
}

register_cpu_operator!{
    Tan,
    UnaryElementwiseOp<TensorTypes<f32>, CPUContext, TanFunctor<CPUContext>>
}

num_inputs!{Tan, 1}

num_outputs!{Tan, 1}

inputs!{Tan, 
    0 => ("input", "Input tensor")
}

outputs!{Tan, 
    0 => ("output", "The tangent of the input tensor computed element-wise")
}

identical_type_and_shape!{Tan}

impl<Context> TanFunctor<Context> {

    pub fn call<T>(
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
           math::Tan(N, X, Y, context);
           return true;
           */
    }
}

pub struct TanGradientFunctor<Context> { 

    phantom: PhantomData<Context>,
}

impl<CPUContext> TanGradientFunctor<CPUContext> {

    pub fn forward<T>(
        x_dims:  &Vec<i32>,
        dY_dims: &Vec<i32>,
        x:       *const T,
        dY:      *const T,
        dX:      *mut T,
        context: *mut CPUContext) -> bool 
    {
        todo!();
        /*
           const int size = std::accumulate(
           X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
           ConstEigenVectorArrayMap<T> dY_arr(dY, size);
           ConstEigenVectorArrayMap<T> X_arr(X, size);
           EigenVectorMap<T>(dX, size) = dY_arr / X_arr.cos().square();
           return true;
           */
    }
}

register_cpu_operator!{
    TanGradient,
    BinaryElementwiseOp<TensorTypes<f32>, CPUContext, TanGradientFunctor<CPUContext>>
}

num_inputs!{TanGradient, 2}

num_outputs!{TanGradient, 1}

identical_type_and_shape!{TanGradient}

pub struct GetTanGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetTanGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
                "TanGradient",
                "",
                std::vector<std::string>{I(0), GO(0)},
                std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Tan, GetTanGradient}
