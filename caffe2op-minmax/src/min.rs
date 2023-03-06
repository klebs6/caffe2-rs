crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MinOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

register_cpu_operator!{Min, MinOp<f32, CPUContext>}

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
