crate::ix!();

/**
  | Element-wise mean of an arbitrary number
  | of input tensors. This operation can
  | be performed in-place, by using the
  | first input blob as the output blob.
  | All inputs must have the same shape and
  | data type, and the output will have the
  | same shape as the inputs.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mean_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MeanOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{Mean, (1,INT_MAX)}

num_outputs!{Mean, 1}

inputs!{Mean, 
    0 => ("X, Y, ...", "*(type: Tensor`<Ord>`)* List of input tensors with the same shape.")
}

outputs!{Mean, 
    0 => ("M", "*(type: Tensor`<Ord>`)* Output tensor with the same dimensions as inputs. Contains the mean values of the input tensors calculated element-wise.")
}

identical_type_and_shape_of_input!{Mean, 0}

allow_inplace!{Mean, vec![(0, 0)]}

impl<Context> MeanOp<Context> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& input0 = Input(0);

        auto* output = Output(0, input0.sizes(), at::dtype<T>());
        output->CopyFrom(input0, true /*async*/);

        if (InputSize() == 1) {
          return true;
        }

        // Dimension checking
        for (int i = 1; i < InputSize(); ++i) {
          if (output->sizes() != Input(i).sizes()) {
            CAFFE_THROW(
                "Check failed: output->sizes() == Input(i).sizes().",
                "Description: Input #",
                i,
                ", input dimension:",
                Input(i).sizes(),
                " should match output dimension: ",
                output->sizes());
          }
        }

        T* output_data = output->template mutable_data<T>();
        for (int i = 1; i < InputSize(); ++i) {
          math::Add(
              output->numel(),
              output_data,
              Input(i).template data<T>(),
              output_data,
              &context_);
        }

        math::Scale(
            output->numel(),
            1.0f / InputSize(),
            output_data,
            output_data,
            &context_);

        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (Input(0).template IsType<float>()) {
          return DoRunWithType<float>();
        } else if (Input(0).template IsType<double>()) {
          return DoRunWithType<double>();
        } else {
          CAFFE_THROW(
              "Mean operator only supports 32-bit float or 64-bit double, but",
              " input was of type ",
              Input(0).dtype().name());
        }
        */
    }
}
