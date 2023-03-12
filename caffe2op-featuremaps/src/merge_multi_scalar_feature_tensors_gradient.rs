crate::ix!();

/**
  | Explode given multi-feature tensors
  | with scalar features into many.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeMultiScalarFeatureTensorsGradientOp<Context> {

    storage:                 OperatorStorage,
    context:                 Context,

    k_num_tensors_per_input: i32, // = 1;
    num_feature_inputs:      i32,
}

inputs!{MergeMultiScalarFeatureTensorsGradient, 
    0 => ("in1_lengths",     ".lengths"),
    1 => ("out_values_grad", ".values_grad")
}

outputs!{MergeMultiScalarFeatureTensorsGradient, 
    0 => ("in1_values_grad", ".values_grad")
}

num_inputs!{MergeMultiScalarFeatureTensorsGradient, 
    |n: i32| {
        n >= 2
    }
}

num_outputs!{MergeMultiScalarFeatureTensorsGradient, 
    |n: i32| {
        n >= 1
    }
}

impl<Context> MergeMultiScalarFeatureTensorsGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        numFeatureInputs_ = (InputSize() - 1) / kNumTensorsPerInput;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
            call(this, Input(InputSize() - 1));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            int numExamples = Input(0).numel();
            std::vector<int> outValuesOffset(numFeatureInputs_);
            for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
              int inputNumValues = 0;
              const int32_t* inLengthsData =
                  Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
              for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
                inputNumValues += inLengthsData[exampleIndex];
              }
              Output(inputIndex)->Resize(inputNumValues);
            }

            const auto& inValuesGrad = Input(InputSize() - 1);
            const T* inValuesGradData = inValuesGrad.template data<T>();

            int inValuesOffset = 0;
            for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
              for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
                const int32_t* inLengthsData =
                    Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
                if (inLengthsData[exampleIndex] > 0) {
                  T* outFeatureValues = Output(inputIndex)->template mutable_data<T>();
                  context_.CopyItemsSameDevice(
                      inValuesGrad.dtype(),
                      inLengthsData[exampleIndex],
                      &inValuesGradData[inValuesOffset],
                      &outFeatureValues[outValuesOffset[inputIndex]]);
                  outValuesOffset[inputIndex] += inLengthsData[exampleIndex];
                  inValuesOffset += inLengthsData[exampleIndex];
                }
              }
            }
            return true;
        */
    }
}

