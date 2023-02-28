crate::ix!();

/**
  | Calculates the hyperbolic tangent
  | of the given input tensor element-wise.
  | 
  | This operation can be done in an in-place
  | fashion too, by providing the same input
  | and output blobs.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tanh_op.cc
  |
  */
pub struct TanhFunctor<Context> { 
    phantom: PhantomData<Context>,
}

num_inputs!{Tanh, 1}

num_outputs!{Tanh, 1}

inputs!{Tanh, 
    0 => ("input", "1-D input tensor")
}

outputs!{Tanh, 
    0 => ("output", "The hyperbolic tangent values of the input tensor, computed element-wise")
}

identical_type_and_shape!{Tanh}

allow_inplace!{Tanh, vec![(0, 0)]}

inherit_onnx_schema!{Tanh}

impl<Context> TanhFunctor<Context> {
    
    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Tanh<T, Context>(N, X, Y, context);
        return true;
        */
    }
}

///--------------------
pub struct TanhGradientFunctor<Context> { 

    phantom: PhantomData<Context>,
}

num_inputs!{TanhGradient, 2}

num_outputs!{TanhGradient, 1}

identical_type_and_shape_of_input!{TanhGradient, 1}

allow_inplace!{TanhGradient, vec![(1, 0)]}

impl<Context> TanhGradientFunctor<Context> {
    
    #[inline] pub fn forward<T>(&self, 
        y_dims:  &Vec<i32>,
        dy_dims: &Vec<i32>,
        y:       *const T,
        dy:      *const T,
        dx:      *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
        
        */
    }
}

#[cfg(caffe2_use_accelerate)]
impl TanhFunctor<CPUContext> {

    #[inline] pub fn invoke_f32(&self, 
        n:       i32,
        x:       *const f32,
        y:       *mut f32,
        context: *mut CPUContext) -> bool {
        
        todo!();
        /*
            vvtanhf(Y, X, &N);
      return true;
        */
    }
}

register_cpu_operator!{
    Tanh,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        TanhFunctor<CPUContext>>
}

#[test] fn tanh_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Tanh",
        ["X"],
        ["X"],
    )

    workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.RunOperatorOnce(op)
    print("X:\n", workspace.FetchBlob("X"))

    ```

    **Result**

    ```

    X:
     [[ 2.032603   -2.3556721  -0.14955314]
     [ 0.39309832 -1.1020128  -0.92951244]
     [-0.62815386  0.21342885  1.4002231 ]]

    X:
     [[ 0.9662601  -0.982175   -0.14844811]
     [ 0.3740282  -0.8012209  -0.73036647]
     [-0.55677974  0.21024609  0.8853999 ]]
    */
}

