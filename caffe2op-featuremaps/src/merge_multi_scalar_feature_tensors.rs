crate::ix!();

/**
  | Merge given multi-feature tensors
  | with scalar features into one.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeMultiScalarFeatureTensorsOp<Context> {

    storage:                 OperatorStorage,
    context:                 Context,

    k_num_tensors_per_input: i32, // = 3;
    num_inputs:              i32,
    in_keys_offset:          Vec<i32>,
}

num_inputs!{MergeMultiScalarFeatureTensors, 
    |n: i32| {
        n >= 3 && n % 3 == 0
    }
}

num_outputs!{MergeMultiScalarFeatureTensors, 3}

inputs!{MergeMultiScalarFeatureTensors, 
    0 => ("in1_lengths", ".lengths"),
    1 => ("in1_keys",    ".keys"),
    2 => ("in1_values",  ".values")
}

outputs!{MergeMultiScalarFeatureTensors, 
    0 => ("out_lengths", ".lengths"),
    1 => ("out_keys",    ".keys"),
    2 => ("out_values",  ".values")
}

impl<Context> MergeMultiScalarFeatureTensorsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        numInputs_ = InputSize() / kNumTensorsPerInput;
        inKeysOffset_.resize(numInputs_);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
            call(this, Input(2));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            int numExamples = Input(0).numel();
            int totalNumFeatures = 0;
            for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
              totalNumFeatures += Input(kNumTensorsPerInput * inputIndex + 1).numel();
            }

            auto* outLengths = Output(0, {numExamples}, at::dtype<int32_t>());
            auto* outKeys = Output(1, {totalNumFeatures}, at::dtype<int64_t>());
            auto* outValues = Output(2, {totalNumFeatures}, at::dtype<T>());

            int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
            int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
            T* outValuesData = outValues->template mutable_data<T>();

            int outKeysOffset = 0;
            for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
              inKeysOffset_[inputIndex] = 0;
            }
            for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
              outLengthsData[exampleIndex] = 0;
              for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
                const int32_t* inLengthsData =
                    Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
                auto inputKeysBlobIdx = kNumTensorsPerInput * inputIndex + 1;
                const int64_t* inKeysData =
                    Input(inputKeysBlobIdx).template data<int64_t>();
                const T* inValuesData =
                    Input(kNumTensorsPerInput * inputIndex + 2).template data<T>();
                outLengthsData[exampleIndex] += inLengthsData[exampleIndex];
                for (int featureIndex = 0; featureIndex < inLengthsData[exampleIndex];
                     ++featureIndex) {
                  CAFFE_ENFORCE_LT(outKeysOffset, totalNumFeatures);
                  CAFFE_ENFORCE_LT(
                      inKeysOffset_[inputIndex], Input(inputKeysBlobIdx).numel());
                  outKeysData[outKeysOffset] = inKeysData[inKeysOffset_[inputIndex]];
                  outValuesData[outKeysOffset] =
                      inValuesData[inKeysOffset_[inputIndex]];
                  ++outKeysOffset;
                  ++inKeysOffset_[inputIndex];
                }
              }
            }

            return true;
        */
    }
}

