crate::ix!();

/**
  | Element-wise application of the ceil
  | function ($y=ceil(x)$) to the input
  | tensor `X`. Output tensor shape is the
  | same as the input tensor.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/ceil_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CeilOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

impl<T,Context> CeilOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        auto* Y = Output(0, X.sizes(), at::dtype<float>());

        const float* Xdata = X.template data<float>();
        float* Ydata = Y->template mutable_data<float>();
        for (int i = 0; i < X.numel(); ++i) {
          Ydata[i] = std::ceil(Xdata[i]);
        }
        return true;
        */
    }
}

register_cpu_operator!{Ceil, CeilOp<f32, CPUContext>}

num_inputs!{Ceil, 1}

num_outputs!{Ceil, 1}

inputs!{Ceil, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Ceil, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor.")
}

allow_inplace!{Ceil, vec![(0, 0)]}

// TODO: Write gradient for this when needed
gradient_not_implemented_yet!{Ceil}
