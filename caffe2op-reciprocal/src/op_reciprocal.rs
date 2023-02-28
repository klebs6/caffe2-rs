crate::ix!();

/**
  | Performs element-wise reciprocal
  | ($\1/x$) of input tensor $X$.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reciprocal_op.cc
  |
  */
pub struct ReciprocalFunctor<Context> {
    
    // Input: X, output: Y
    phantom: PhantomData<Context>,

}

num_inputs!{Reciprocal, 1}

num_outputs!{Reciprocal, 1}

inputs!{Reciprocal, 
    0 => ("X", "*(type: Tensor`<f32>`)* Input data tensor.")
}

outputs!{Reciprocal, 
    0 => ("Y", "*(type: Tensor`<f32>`)* Output tensor.")
}

identical_type_and_shape!{Reciprocal}

allow_inplace!{Reciprocal, vec![(0, 0)]}

#[test] fn reciprocal_functor_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Reciprocal",
        ["X"],
        ["Y"],
    )

    workspace.FeedBlob("X", (np.random.randint(10, size=(3,3))).astype(np.f3232))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[8. 3. 3.]
     [4. 0. 0.]
     [1. 2. 5.]]
    Y:
    [[0.125 0.3333333  0.3333333 ]
     [0.25  inf        inf       ]
     [1     0.5        0.2       ]]
    */
}

impl<Context> ReciprocalFunctor<Context> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Inv(N, X, Y, context);
        return true;
        */
    }
}

///----------------------------------
pub struct ReciprocalGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{ReciprocalGradient, 2}

num_outputs!{ReciprocalGradient, 1}

allow_inplace!{ReciprocalGradient, vec![(1, 0)]}

impl<CPUContext> ReciprocalGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        y_dims:  &Vec<i32>,
        dY_dims: &Vec<i32>,
        y:       *const T,
        dY:      *const T,
        dX:      *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
          const int size = std::accumulate(
              Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> Y_arr(Y, size);
          EigenVectorMap<T>(dX, size) = dY_arr * (-Y_arr.square());
          return true;
        */
    }
}

register_cpu_operator!{Reciprocal,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        ReciprocalFunctor<CPUContext>>}


register_cpu_operator!{ReciprocalGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        ReciprocalGradientFunctor<CPUContext>>}

pub struct GetReciprocalGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReciprocalGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ReciprocalGradient",
            "",
            std::vector<std::string>{O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Reciprocal, GetReciprocalGradient}
