crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeSingleListOrMapFeatureTensorsGradientOp<Context> {
    storage:               OperatorStorage,
    context:               Context,
    num_tensors_per_input: i32,//2
    num_feature_inputs:    i32,
}

impl<Context> MergeSingleListOrMapFeatureTensorsGradientOp<Context> {
    
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
              const bool* inPresenceData =
                  Input(kNumTensorsPerInput * inputIndex + 1).template data<bool>();
              for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
                if (inPresenceData[exampleIndex]) {
                  inputNumValues += inLengthsData[exampleIndex];
                }
              }
              Output(inputIndex)->Resize(inputNumValues);
            }

            const auto& inValuesValuesGrad = Input(InputSize() - 1);
            const T* inValuesValuesGradData = inValuesValuesGrad.template data<T>();

            int inValuesValuesOffset = 0;
            for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
              for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
                const int32_t* inLengthsData =
                    Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
                const bool* inPresenceData =
                    Input(kNumTensorsPerInput * inputIndex + 1).template data<bool>();
                if (inPresenceData[exampleIndex]) {
                  T* outFeatureValues = Output(inputIndex)->template mutable_data<T>();
                  context_.CopyItemsSameDevice(
                      inValuesValuesGrad.dtype(),
                      inLengthsData[exampleIndex],
                      &inValuesValuesGradData[inValuesValuesOffset],
                      &outFeatureValues[outValuesOffset[inputIndex]]);
                  outValuesOffset[inputIndex] += inLengthsData[exampleIndex];
                  inValuesValuesOffset += inLengthsData[exampleIndex];
                }
              }
            }
            return true;
        */
    }
}
