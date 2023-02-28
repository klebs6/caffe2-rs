crate::ix!();

/**
  | Performs element-wise square-root
  | ($\sqrt{x}$) of input tensor $X$.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqrt_op.cc
  |
  */
pub struct SqrtFunctor<Context> { 

    /**
      | Input: X,
      | 
      | output: Y
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{Sqrt, 1}

num_outputs!{Sqrt, 1}

inputs!{Sqrt, 
    0 => ("X", "*(type: Tensor`<float>`)* Input data tensor.")
}

outputs!{Sqrt, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor.")
}

identical_type_and_shape!{Sqrt}

allow_inplace!{Sqrt, vec![(0, 0)]}

impl<Context> SqrtFunctor<Context> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Sqrt<T, Context>(N, X, Y, context);
        return true;
        */
    }
}

register_cpu_operator!{Sqrt,
    UnaryElementwiseOp<
        TensorTypes<f32, f64>,
        CPUContext,
        SqrtFunctor<CPUContext>>}

#[test] fn sqrt_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Sqrt",
        ["X"],
        ["Y"],
    )

    workspace.FeedBlob("X", (np.random.randint(10, size=(3,3))).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[8. 3. 3.]
     [4. 0. 0.]
     [1. 2. 5.]]
    Y:
    [[2.8284268  1.7320508  1.7320508 ]
     [1.9999999  0.         0.        ]
     [0.99999994 1.4142134  2.236068  ]]
    */
}

pub struct GetSqrtGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSqrtGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            Argument scale_arg;
        scale_arg.set_name("scale");
        scale_arg.set_f(0.5);
        return std::vector<OperatorDef>{CreateOperatorDef(
                                            "Scale",
                                            "",
                                            std::vector<std::string>{GO(0)},
                                            std::vector<std::string>{GI(0)},
                                            std::vector<Argument>{scale_arg}),
                                        CreateOperatorDef(
                                            "Div",
                                            "",
                                            std::vector<std::string>{GI(0), O(0)},
                                            std::vector<std::string>{GI(0)})};
        */
    }
}

register_gradient!{Sqrt, GetSqrtGradient}

register_cuda_operator!{Sqrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        SqrtFunctor<CUDAContext>>
}
