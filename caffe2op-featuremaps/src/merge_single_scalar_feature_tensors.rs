crate::ix!();

/**
  | Merge given single-feature tensors
  | with scalar features into one multi-feature
  | tensor.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeSingleScalarFeatureTensorsOp<Context> {
    storage: OperatorStorage,
    context: Context,

    num_tensors_per_input: i32, // 2
    num_inputs:  i32,
    feature_ids: Vec<i64>,
}

num_inputs!{MergeSingleScalarFeatureTensors, 
    |n: i32| {
        n >= 2 && n % 2 == 0
    }
}

num_outputs!{MergeSingleScalarFeatureTensors, 3}

inputs!{MergeSingleScalarFeatureTensors, 
    0 => ("in1",          ""),
    1 => ("in1_presence", ".presence")
}

outputs!{MergeSingleScalarFeatureTensors, 
    0 => ("out_lengths", ".lengths"),
    1 => ("out_keys",    ".keys"),
    2 => ("out_values",  ".values")
}

args!{MergeSingleScalarFeatureTensors, 
    0 => ("feature_ids", "feature ids")
}

impl<Context> MergeSingleScalarFeatureTensorsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        numInputs_ = InputSize() / kNumTensorsPerInput;
        featureIDs_ = this->template GetRepeatedArgument<int64_t>("feature_ids");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
            call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            int numExamples = Input(0).numel();
            int totalNumFeatures = 0;
            for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
              const bool* inPresenceData =
                  Input(kNumTensorsPerInput * inputIndex + 1).template data<bool>();
              for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
                if (inPresenceData[exampleIndex]) {
                  ++totalNumFeatures;
                }
              }
            }

            auto* outLengths = Output(0, {numExamples}, at::dtype<int32_t>());
            auto* outKeys = Output(1, {totalNumFeatures}, at::dtype<int64_t>());
            auto* outValues = Output(2, {totalNumFeatures}, at::dtype<T>());

            int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
            int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
            T* outValuesData = outValues->template mutable_data<T>();

            int keysOffset = 0;
            for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
              outLengthsData[exampleIndex] = 0;
              for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
                const T* inData =
                    Input(kNumTensorsPerInput * inputIndex).template data<T>();
                const bool* inPresenceData =
                    Input(kNumTensorsPerInput * inputIndex + 1).template data<bool>();
                if (inPresenceData[exampleIndex]) {
                  ++outLengthsData[exampleIndex];
                  outKeysData[keysOffset] = featureIDs_[inputIndex];
                  outValuesData[keysOffset] = inData[exampleIndex];
                  ++keysOffset;
                }
              }
            }
            return true;
        */
    }
}

