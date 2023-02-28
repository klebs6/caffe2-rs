crate::ix!();

pub struct SinFunctor<Context> {
    
    phantom: PhantomData<Context>,
}

impl<Context> SinFunctor<Context> {
    
    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
        math::Sin(N, X, Y, context);
        return true;
        */
    }
}

///-------------------
pub struct SinGradientFunctor<Context> {
    
    phantom: PhantomData<Context>,
}

impl SinGradientFunctor<CPUContext> {
    
    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        dy_dims: &Vec<i32>,
        x:       *const T,
        dy:      *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
          const int size = std::accumulate(
          X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> X_arr(X, size);
          EigenVectorMap<T>(dX, size) = dY_arr * X_arr.cos();
          return true;
        */
    }
}

/**
  | Calculates the sine of the given input
  | tensor, element-wise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sin_op.cc
  |
  */
register_cpu_operator!{Sin,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SinFunctor<CPUContext>>}

num_inputs!{Sin, 1}

num_outputs!{Sin, 1}

inputs!{Sin, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Sin, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor calculated as the sine of the input tensor, element-wise.")
}

identical_type_and_shape!{Sin}

#[test] fn sin_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Sin",
        ["X"],
        ["Y"]
    )

    workspace.FeedBlob("X", np.random.rand(5).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))


    X: [0.8466114  0.1803606  0.5601509  0.04959291 0.64770824]
    Y: [0.74903965 0.17938434 0.5313141  0.04957259 0.60336035]
    */
}

///-----------------------------
register_cpu_operator!{SinGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SinGradientFunctor<CPUContext>>}

num_inputs!{SinGradient, 2}

num_outputs!{SinGradient, 1}

identical_type_and_shape!{SinGradient}

pub struct GetSinGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSinGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SinGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Sin, GetSinGradient}
