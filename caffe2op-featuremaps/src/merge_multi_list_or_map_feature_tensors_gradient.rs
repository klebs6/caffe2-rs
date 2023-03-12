crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeMultiListOrMapFeatureTensorsGradientOp<Context> {
    storage:                 OperatorStorage,
    context:                 Context,
    k_num_tensors_per_input: i32, // = 2;
    num_feature_inputs:      i32,
}

impl<Context> MergeMultiListOrMapFeatureTensorsGradientOp<Context> {
    
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
            std::vector<int> outValuesLengthOffset(numFeatureInputs_);
            std::vector<int> outValuesValuesOffset(numFeatureInputs_);
            for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
              int inputNumValues = 0;
              auto& inValuesLength = Input(kNumTensorsPerInput * inputIndex + 1);
              const int32_t* inValuesLengthsData =
                  inValuesLength.template data<int32_t>();
              for (int valuesIndex = 0; valuesIndex < inValuesLength.numel();
                   ++valuesIndex) {
                inputNumValues += inValuesLengthsData[valuesIndex];
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
                const int32_t* inValuesLengthsData =
                    Input(kNumTensorsPerInput * inputIndex + 1)
                        .template data<int32_t>();
                int valuesLengthCopy = 0;
                for (int valuesLengthIndex = 0;
                     valuesLengthIndex < inLengthsData[exampleIndex];
                     ++valuesLengthIndex) {
                  valuesLengthCopy += inValuesLengthsData
                      [outValuesLengthOffset[inputIndex] + valuesLengthIndex];
                }
                if (valuesLengthCopy > 0) {
                  T* outFeatureValues = Output(inputIndex)->template mutable_data<T>();
                  context_.CopyItemsSameDevice(
                      inValuesValuesGrad.dtype(),
                      valuesLengthCopy,
                      &inValuesValuesGradData[inValuesValuesOffset],
                      &outFeatureValues[outValuesValuesOffset[inputIndex]]);
                }
                outValuesLengthOffset[inputIndex] += inLengthsData[exampleIndex];
                outValuesValuesOffset[inputIndex] += valuesLengthCopy;
                inValuesValuesOffset += valuesLengthCopy;
              }
            }
            return true;
        */
    }
}
