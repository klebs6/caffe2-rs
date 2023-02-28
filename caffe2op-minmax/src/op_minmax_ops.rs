crate::ix!();

use crate::{
    OperatorDef,
    OperatorStorage,
    GradientMakerBase,
};

///---------------------------------
#[test] fn max_op_example() {

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Max",
        ["X", "Y", "Z"],
        ["X"],
    )

    workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32))
    workspace.FeedBlob("Y", (np.random.rand(3,3)).astype(np.float32))
    workspace.FeedBlob("Z", (np.random.rand(3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    print("Y:", workspace.FetchBlob("Y"))
    print("Z:", workspace.FetchBlob("Z"))
    workspace.RunOperatorOnce(op)
    print("Max:", workspace.FetchBlob("X"))

    X:
    [[0.4496477  0.07061381 0.7139333 ]
     [0.83203    0.05970785 0.72786295]
     [0.75988126 0.04601283 0.32820013]]
    Y:
    [[0.05683139 0.16872478 0.671098  ]
     [0.70739156 0.09878621 0.03416285]
     [0.34087983 0.94986707 0.67263436]]
    Z:
    [[0.48051122 0.07141234 0.85264146]
     [0.77086854 0.22082241 0.13154659]
     [0.42401117 0.995431   0.4263775 ]]
    Max:
    [[0.48051122 0.16872478 0.85264146]
     [0.83203    0.22082241 0.72786295]
     [0.75988126 0.995431   0.67263436]]

    */
}

/**
  | Element-wise max of an arbitrary number
  | of input tensors.
  | 
  | This operation can be performed in-place,
  | by using the first input blob as the output
  | blob. All inputs must have the same shape
  | and data type, and the output will have
  | the same shape as the inputs.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/minmax_ops.cc
  |
  */
pub struct MaxOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{Max, (1,INT_MAX)}

num_outputs!{Max, 1}

inputs!{Max, 
    0 => ("X, Y, ...", "*(type: Tensor`<Ord>`)* List of input tensors with the same shape.")
}

outputs!{Max, 
    0 => ("M", "*(type: Tensor`<Ord>`)* Output tensor with same dimensions as input(s). Contains the maximum valued element at each location.")
}

identical_type_and_shape_of_input!{Max, 0}

allow_inplace!{Max, vec![(0, 0)]}

inherit_onnx_schema!{Max}

impl<T,Context> MaxOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X0 = Input(0);
        auto* Y = Output(0);
        Y->ResizeLike(X0);
        const T* X0_data = X0.template data<T>();
        T* Y_data = Y->template mutable_data<T>();
        const int N = X0.numel();
        if (InputSize() == 1) {
          if (Y != &X0) {
            context_.template CopySameDevice<T>(N, X0_data, Y_data);
          }
          return true;
        }
        const auto& X1 = Input(1);
        CAFFE_ENFORCE_EQ(
            X0.sizes(),
            Y->sizes(),
            "Description: Input #1, input dimension:",
            X1.sizes(),
            " should match output dimension: ",
            Y->sizes());
        const T* X1_data = X1.template data<T>();
        math::Max<T, Context>(N, X0_data, X1_data, Y_data, &context_);
        for (int i = 2; i < InputSize(); ++i) {
          const auto& Xi = Input(i);
          CAFFE_ENFORCE_EQ(
              Xi.sizes(),
              Y->sizes(),
              "Description: Input #",
              i,
              ", input dimension:",
              Input(i).sizes(),
              " should match output dimension: ",
              Y->sizes());
          const T* Xi_data = Xi.template data<T>();
          math::Max<T, Context>(N, Y_data, Xi_data, Y_data, &context_);
        }
        return true;
        */
    }
}

///------------------------------
#[test] fn min_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Min",
        ["X", "Y", "Z"],
        ["X"],
    )

    workspace.FeedBlob("X", (np.random.rand(2,2)).astype(np.float32))
    workspace.FeedBlob("Y", (np.random.rand(2,2)).astype(np.float32))
    workspace.FeedBlob("Z", (np.random.rand(2,2)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    print("Y:", workspace.FetchBlob("Y"))
    print("Z:", workspace.FetchBlob("Z"))
    workspace.RunOperatorOnce(op)
    print("Min:", workspace.FetchBlob("X"))

    X:
    [[0.32731926 0.4939747 ]
     [0.29242373 0.43460014]]
    Y:
    [[0.40928316 0.916115  ]
     [0.77526504 0.29339448]]
    Z:
    [[0.7899794  0.90335774]
     [0.82599413 0.2843068 ]]
    Min:
    [[0.32731926 0.4939747 ]
     [0.29242373 0.2843068 ]]

    */
}

/**
  | Element-wise min of an arbitrary number
  | of input tensors.
  | 
  | This operation can be performed in-place,
  | by using the first input blob as the output
  | blob. All inputs must have the same shape
  | and data type, and the output will have
  | the same shape as the inputs.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/minmax_ops.cc
  |
  */
pub struct MinOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{Min, (1,INT_MAX)}

num_outputs!{Min, 1}

inputs!{Min, 
    0 => ("X, Y, ...", "*(type: Tensor`<Ord>`)* List of input tensors with the same shape.")
}

outputs!{Min, 
    0 => ("M", "*(type: Tensor`<Ord>`)* Output tensor with same dimensions as input(s). Contains the minimum valued element at each location.")
}

identical_type_and_shape_of_input!{Min, 0}

allow_inplace!{Min, vec![(0, 0)]}

inherit_onnx_schema!{Min}

impl<T,Context> MinOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X0 = Input(0);
        auto* Y = Output(0);
        Y->ResizeLike(X0);
        const T* X0_data = X0.template data<T>();
        T* Y_data = Y->template mutable_data<T>();
        const int N = X0.numel();
        if (InputSize() == 1) {
          if (Y != &X0) {
            context_.template CopySameDevice<T>(N, X0_data, Y_data);
          }
          return true;
        }
        const auto& X1 = Input(1);
        CAFFE_ENFORCE_EQ(
            X0.sizes(),
            Y->sizes(),
            "Description: Input #1, input dimension:",
            X1.sizes(),
            " should match output dimension: ",
            Y->sizes());
        const T* X1_data = X1.template data<T>();
        math::Min<T, Context>(N, X0_data, X1_data, Y_data, &context_);
        for (int i = 2; i < InputSize(); ++i) {
          const auto& Xi = Input(i);
          CAFFE_ENFORCE_EQ(
              Xi.sizes(),
              Y->sizes(),
              "Description: Input #",
              i,
              ", input dimension:",
              Input(i).sizes(),
              " should match output dimension: ",
              Y->sizes());
          const T* Xi_data = Xi.template data<T>();
          math::Min<T, Context>(N, Y_data, Xi_data, Y_data, &context_);
        }
        return true;
        */
    }
}

///-------------------------------
pub struct SelectGradientOpBase<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

///-------------------------------
pub struct MaxGradientOp<T,Context> {
    base: SelectGradientOpBase<T, Context>,
    phantom: PhantomData<T>,
}

num_inputs!{MaxGradient, (3,INT_MAX)}

num_outputs!{MaxGradient, (1,INT_MAX)}

impl<T,Context> MaxGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : SelectGradientOpBase<T, Context>(std::forward<Args>(args)...)
        */
    }
}

///--------------------------------
pub struct MinGradientOp<T,Context> {
    base: SelectGradientOpBase<T, Context>,
    phantom: PhantomData<T>,
}

num_inputs!{MinGradient, (3,INT_MAX)}

num_outputs!{MinGradient, (1,INT_MAX)}

impl<T,Context> MinGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : SelectGradientOpBase<T, Context>(std::forward<Args>(args)...)
        */
    }
}

register_cpu_operator!{Min, MinOp<f32, CPUContext>}

register_cpu_operator!{Max, MaxOp<f32, CPUContext>}

impl<T, Context> SelectGradientOpBase<T, Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& Y = Input(0);
      const auto& dY = Input(1);
      const int N = Y.numel();
      ConstEigenVectorArrayMap<T> Y_arr(Y.template data<T>(), N);
      ConstEigenVectorArrayMap<T> dY_arr(dY.template data<T>(), N);
      for (int i = 0; i < OutputSize(); i++) {
        const auto& Xi = Input(i + 2);
        auto* dXi = Output(i, Xi.sizes(), at::dtype<T>());
        ConstEigenVectorArrayMap<T> Xi_arr(Xi.template data<T>(), N);
        EigenVectorArrayMap<T> dXi_arr(dXi->template mutable_data<T>(), N);
        dXi_arr = (Xi_arr == Y_arr).template cast<T>() * dY_arr;
      }
      return true;
        */
    }
}

register_cpu_operator!{MaxGradient, MaxGradientOp<float, CPUContext>}

register_cpu_operator!{MinGradient, MinGradientOp<float, CPUContext>}

pub struct GetMaxGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMaxGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            std::vector<std::string> inputs = {O(0), GO(0)};
        std::vector<std::string> grad_inputs;
        for (int i = 0; i < def_.input_size(); ++i) {
          inputs.push_back(I(i));
          grad_inputs.push_back(GI(i));
        }
        return SingleGradientDef("MaxGradient", "", inputs, grad_inputs);
        */
    }
}

pub struct GetMinGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMinGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            std::vector<std::string> inputs = {O(0), GO(0)};
        std::vector<std::string> grad_inputs;
        for (int i = 0; i < def_.input_size(); ++i) {
          inputs.push_back(I(i));
          grad_inputs.push_back(GI(i));
        }
        return SingleGradientDef("MinGradient", "", inputs, grad_inputs);
        */
    }
}

register_gradient!{Max, GetMaxGradient}

register_gradient!{Min, GetMinGradient}
