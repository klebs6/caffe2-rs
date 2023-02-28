crate::ix!();

/**
  | Softplus takes one input data tensor
  | $X$ and produces one output data tensor
  | $Y,$ where the softplus function, $y
  | = ln(e^x + 1)$, is applied to $X$ elementwise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.cc
  |
  */
pub struct SoftplusOp<DataType> {

    storage:         OperatorStorage,
    context:         CPUContext,

    /**
      | Input: X
      | 
      | output: Y
      |
      */
    phantomDataType: PhantomData<DataType>,
}

num_inputs!{Softplus, 1}

num_outputs!{Softplus, 1}

inputs!{Softplus, 
    0 => ("X", "Input data blob to be operated on.")
}

outputs!{Softplus, 
    0 => ("Y", "Output data blob with same shape as input.")
}

identical_type_and_shape!{Softplus}

allow_inplace!{Softplus, vec![(0, 0)]}

inherit_onnx_schema!{Softplus}

impl SoftplusOp<DataType> {

    fn run_on_device() -> bool {

        todo!();
        /*
        auto& X = Input(0);

        auto* Y = Output(0, X.sizes(), at::dtype<float>());

        EigenVectorMap<float>(Y->template mutable_data<float>(), X.numel()) =
            (ConstEigenVectorMap<float>(X.data<float>(), X.numel()).array().exp() +
             1.0f)
            .log();
        return true;
        */
    }
}

///-----------------------------------------
pub struct SoftplusGradientOp<DataType> {

    storage:         OperatorStorage,
    context:         CPUContext,

    /**
      | Input: Y, dY,
      | 
      | output: dX
      |
      */
    phantomDataType: PhantomData<DataType>,
}

num_inputs!{SoftplusGradient, 2}

num_outputs!{SoftplusGradient, 1}

allow_inplace!{SoftplusGradient, vec![(1, 0)]}

impl<DataType> SoftplusGradientOp<DataType> {

    fn run_on_device() -> bool {
        todo!();
        /*
          auto& Y = Input(0);
          auto& dY = Input(1);

          DCHECK_EQ(dY.numel(), Y.numel());
          auto* dX = Output(0, Y.sizes(), at::dtype<float>());

          const float* Ydata = Y.data<float>();
          const float* dYdata = dY.data<float>();
          float* dXdata = dX->template mutable_data<float>();
          EigenVectorArrayMap<float> dXvec(dXdata, dX->numel());
          ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.numel());
          ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.numel());
          dXvec = dYvec * (1.0 - (-Yvec).exp());
          return true;
        */
    }
}

register_cpu_operator!{Softplus,         SoftplusOp<f32, CPUContext>}

register_cpu_operator!{SoftplusGradient, SoftplusGradientOp<f32, CPUContext>}

#[test] fn softplus_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Softplus",
        ["X"],
        ["Y"],
    )

    workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[-0.5380011   0.65190786  0.55673236]
     [-0.16272168  0.5451048   0.30880353]
     [-0.76606876 -0.6238556  -0.40444514]]

    Y:
     [[0.4598992  1.0713093  1.0097669 ]
     [0.61509246 1.0023911  0.8594219 ]
     [0.38174385 0.42909983 0.5112337 ]]

    */
}

pub struct GetSoftplusGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSoftplusGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SoftplusGradient",
            "",
            vector<string>{O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{Softplus, GetSoftplusGradient}

