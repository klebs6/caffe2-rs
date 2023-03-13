crate::ix!();

/**
  | The *HasElements* op accepts a single
  | or multiple input tensors, and produces
  | a single boolean output $has\_elements$.
  | The output is *True* if and only if any
  | of the input tensor has size > 0.
  | 
  | Note, this op is the opposite of the *IsEmpty*
  | op.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct HasElementsOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

should_not_do_gradient!{IsEmpty}

should_not_do_gradient!{HasElements}

num_inputs!{HasElements, (1,INT_MAX)}

num_outputs!{HasElements, 1}

inputs!{HasElements, 
    0 => ("X1, X2, ...", "List of input data tensors to check for elements.")
}

outputs!{HasElements, 
    0 => ("has_elements", "Output scalar boolean tensor. True if input has size > 0.")
}

impl<Context> HasElementsOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            bool res = false;
        for (auto i = 0; i < InputSize(); ++i) {
          const auto& input = Input(i);
          res = res || input.numel() > 0;
        }
        auto* output = Output(0);
        output->Resize(std::vector<int64_t>{});
        *output->template mutable_data<bool>() = res;
        return true;
        */
    }
}
