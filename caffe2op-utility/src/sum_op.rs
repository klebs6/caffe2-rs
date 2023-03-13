crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SumOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{SumInt, (1,INT_MAX)}

num_outputs!{SumInt, 1}

inputs_can_cross_devices!{SumInt}

tensor_inference_function!{SumInt, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out(1);
          out.push_back(in[0]);
          out[0].set_data_type(TensorProto::INT32);
          return out;
        */
    }
}

allow_inplace!{SumInt, vec![(0, 0)]}

impl<Context> SumOp<Context> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self, ) -> bool {
        todo!();
        /*
            auto& input0 = Input(0);

        if (InputSize() == 1) {
          // TODO: better TensorOptions argument passing(e.g. default argument)
          OutputTensorCopyFrom(
              0,
              // I'll change the order of argument in another diff, so that we don't
              // need to write this
              at::dtype(input0.dtype()),
              input0,
              true /*async*/);
          return true;
        }
        auto* output = Output(0, input0.sizes(), at::dtype<T>());
        T* output_data = output->template mutable_data<T>();
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

        // Add the first two - works if in-place or not.
        math::Add(
            output->numel(),
            input0.template data<T>(),
            Input(1).template data<T>(),
            output_data,
            &context_);
        // Add remaining.
        for (int i = 2; i < InputSize(); ++i) {
          math::Add(
              output->numel(),
              output_data,
              Input(i).template data<T>(),
              output_data,
              &context_);
        }
        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double, int32_t, int64_t>>::call(
            this, Input(0));
        */
    }
}
