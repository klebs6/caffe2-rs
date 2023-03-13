crate::ix!();

/**
  | WeightedSumOp computes the weighted
  | sum of several tensors.
  | 
  | The input should be in the form X_0, weight_0,
  | X_1, weight_1, ... where X_i all have
  | the same shape, and weight_i are size
  | 1 tensors that specifies the weight
  | of each vector.
  | 
  | -----------
  | @note
  | 
  | if one wants to do in-place computation,
  | it could only be done with X_0 also as
  | the output, but not other X_i.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WeightedSumOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_outputs!{WeightedSum, 1}

inputs!{WeightedSum, 
    0 => ("data_0", "First of the input tensors."),
    1 => ("weight_0", "Weight of the first input in the sum.")
}

outputs!{WeightedSum, 
    0 => ("output", "Result containing weighted elem-wise sum of inputs.")
}

num_inputs!{WeightedSum, 
    |n: i32| {
        n > 0 && n % 2 == 0
    }
}

tensor_inference_function!{WeightedSum, 
    WeightedSumShapeInference 
}

cost_inference_function!{WeightedSum, 
    CostInferenceForWeightedSum 
}

allow_inplace!{WeightedSum, vec![(0, 0)]}

identical_type_and_shape_of_input!{WeightedSum, 0}

impl<Context> WeightedSumOp<Context> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            // the code is written this way because of 10.1 + gcc 7.3.1 compiler bug
        // as discussed at
        // https://devtalk.nvidia.com/default/topic/1048037/linux/cuda-10-1-nvidia-you-re-now-quot-fixing-quot-gcc-bugs-that-gcc-doesn-t-even-have/
        const int input_size = (*this).InputSize();
        CAFFE_ENFORCE_EQ(input_size % 2, 0);
        const auto& X0 = Input(0);
        const auto& weight0 = Input(1);
        CAFFE_ENFORCE_EQ(weight0.numel(), 1);
        const int size = X0.numel();
        // Note: removed Aliasing check, since Output already has
        // caching capability
        auto* Y = Output(0, X0.sizes(), at::dtype<T>());
        T* Y_data = Y->template mutable_data<T>();
        if (X0.numel() == 0) {
          return true;
        }
        CAFFE_ENFORCE_GT(X0.numel(), 0);
        if (input_size == 2) {
          math::Scale<float, T>(
              size,
              weight0.template data<float>(),
              X0.template data<T>(),
              Y_data,
              &context_);
          return true;
        }
        const auto& X1 = Input(2);
        CAFFE_ENFORCE(
            !IsInputOutputAlias(2, 0),
            "Input #2 is the same as output. If you want to do in-place updates, "
            "put the output as input #0.");
        const auto& weight1 = Input(3);
        CAFFE_ENFORCE_EQ(X1.numel(), size);
        CAFFE_ENFORCE_EQ(weight1.numel(), 1);
        if (!IsInputOutputAlias(0, 0)) {
          context_.template CopySameDevice<T>(size, X0.template data<T>(), Y_data);
        }
        math::Axpby<float, T, Context>(
            size,
            weight1.template data<float>(),
            X1.template data<T>(),
            weight0.template data<float>(),
            Y_data,
            &context_);
        for (int i = 4; i < input_size; i += 2) {
          const auto& Xi = Input(i);
          // Do a check: if the input is the same as output, we have a problem -
          // in-place update should always only happen with the zeroth input.
          const std::string err_msg = "Input #" + to_string(i) +
              " is the same as output. If you want to do in-place updates, "
              "put the output as input #0.";
          CAFFE_ENFORCE(!IsInputOutputAlias(i, 0), err_msg);
          const auto& weighti = Input(i + 1);
          CAFFE_ENFORCE_EQ(Xi.numel(), size);
          CAFFE_ENFORCE_EQ(weighti.numel(), 1);
          math::Axpy<float, T, Context>(
              size,
              weighti.template data<float>(),
              Xi.template data<T>(),
              Y_data,
              &context_);
        }
        return true;
        */
    }
}

