crate::ix!();

/**
  | SwishGradient takes X, Y and dY and uses
  | this to update dX according to the chain
  | rule and derivatives of the swish function.
  |
  */
pub struct SwishGradientOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    /*
      | Input: X, Y, dY,
      | 
      | output: dX
      |
      */
}

num_inputs!{SwishGradient, 3}

num_outputs!{SwishGradient, 1}

allow_inplace!{SwishGradient, vec![(2, 0)]}

impl<Context> SwishGradientOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(X));
        */
    }
}

input_tags!{
    SwishGradientOp {
        X,
        Y,
        Dy
    }
}

output_tags!{
    SwishGradientOp {
        Dx
    }
}

/**
  | Swish takes one input data (Tensor)
  | and produces one output data (Tensor)
  | where the swish function, y = x / (1 + exp(-x)),
  | is applied to the tensor elementwise.
  |
  */
pub struct SwishFunctor<T, Context> {

    /**
      | Input: X,
      | 
      | output: Y
      |
      */
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Context>,
}

num_inputs!{Swish, 1}

num_outputs!{Swish, 1}

inputs!{Swish, 
    0 => ("X", "1D input tensor")
}

outputs!{Swish, 
    0 => ("Y", "1D output tensor")
}

identical_type_and_shape!{Swish}

impl<T, CPUContext> SwishFunctor<T, CPUContext> {

    #[inline] pub fn invoke(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            ConstEigenVectorArrayMap<T> X_arr(X, N);
                EigenVectorArrayMap<T>(Y, N) = X_arr / (T(1) + (-X_arr).exp());
                return true;
        */
    }
}

impl<CPUContext> SwishGradientOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        
        todo!();
        /*
            auto& Xin = Input(X);
                auto& Yin = Input(Y);
                auto& DYin = Input(DY);

                CAFFE_ENFORCE_EQ(Xin.numel(), Yin.numel());
                CAFFE_ENFORCE_EQ(DYin.numel(), Yin.numel());
                auto* DXout = Output(DX, Yin.sizes(), at::dtype<float>());

                const float* Xdata = Xin.template data<float>();
                const float* Ydata = Yin.template data<float>();
                const float* dYdata = DYin.template data<float>();
                float* dXdata = DXout->template mutable_data<float>();

                EigenVectorArrayMap<float> dXvec(dXdata, DXout->numel());
                ConstEigenVectorArrayMap<float> Xvec(Xdata, Xin.numel());
                ConstEigenVectorArrayMap<float> Yvec(Ydata, Yin.numel());
                ConstEigenVectorArrayMap<float> dYvec(dYdata, DYin.numel());

                // dx = dy * (y + sigmoid(x)*(1-y))
                dXvec = dYvec * (Yvec + (T(1) / (T(1) + (-Xvec).exp())) * (T(1) - Yvec));
                return true;
        */
    }
}

register_cpu_operator!{Swish,
    UnaryElementwiseOp<
    TensorTypes<f32>,
    CPUContext,
    SwishFunctor<CPUContext>>
}

register_cpu_operator!{SwishGradient, SwishGradientOp<CPUContext>}

pub struct GetSwishGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSwishGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
                        "SwishGradient",
                        "",
                        std::vector<std::string>{I(0), O(0), GO(0)},
                        std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Swish, GetSwishGradient}
