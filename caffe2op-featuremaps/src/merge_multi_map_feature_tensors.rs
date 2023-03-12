crate::ix!();

/**
  | Merge given multi-feature tensors
  | with map features into one.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeMultiMapFeatureTensorsOp<Context> {

    storage:                 OperatorStorage,
    context:                 Context,

    k_num_tensors_per_input: i32, // = 5;
    num_inputs:              i32,
    in_keys_offset:          Vec<i32>,
    in_values_values_offset: Vec<i32>,
}

num_outputs!{MergeMultiMapFeatureTensors, 5}

num_inputs!{MergeMultiMapFeatureTensors, 
    |n: i32| {
        n >= 5 && n % 5 == 0
    }
}

inputs!{MergeMultiMapFeatureTensors, 
    0 => ("in1_lengths",         ".lengths"),
    1 => ("in1_keys",            ".keys"),
    2 => ("in1_values_lengths",  ".values.lengths"),
    3 => ("in1_values_keys",     ".values.keys"),
    4 => ("in1_values_values",   ".values.values")
}

outputs!{MergeMultiMapFeatureTensors, 
    0 => ("out_lengths",         ".lengths"),
    1 => ("out_keys",            ".keys"),
    2 => ("out_values_lengths",  ".values_lengths"),
    3 => ("out_values_keys",     ".values.keys"),
    4 => ("out_values_values",   ".values.values")
}

impl<Context> MergeMultiMapFeatureTensorsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        numInputs_ = InputSize() / kNumTensorsPerInput;
        inKeysOffset_.resize(numInputs_);
        inValuesValuesOffset_.resize(numInputs_);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
            call(this, Input(3));
        */
    }

    #[inline] pub fn do_run_with_type<K>(&mut self) -> bool {
        todo!();
        /*
            return DispatchHelper<
                TensorTypes2<bool, int32_t, int64_t, float, double, std::string>,
                K>::call(this, Input(4));
        */
    }

    #[inline] pub fn do_run_with_type2<K, V>(&mut self) -> bool {
        todo!();
        /*
            int numExamples = Input(0).numel();
            int totalNumFeatures = 0;
            int totalNumValues = 0;
            for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
              totalNumFeatures += Input(kNumTensorsPerInput * inputIndex + 1).numel();
              totalNumValues += Input(kNumTensorsPerInput * inputIndex + 4).numel();
            }

            auto* outLengths = Output(0, {numExamples}, at::dtype<int32_t>());
            auto* outKeys = Output(1, {totalNumFeatures}, at::dtype<int64_t>());
            auto* outValuesLengths =
                Output(2, {totalNumFeatures}, at::dtype<int32_t>());
            auto* outValuesKeys = Output(3, {totalNumValues}, at::dtype<K>());
            auto* outValuesValues = Output(4, {totalNumValues}, at::dtype<V>());

            int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
            int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
            int32_t* outValuesLengthsData =
                outValuesLengths->template mutable_data<int32_t>();
            K* outValuesKeysData = outValuesKeys->template mutable_data<K>();
            V* outValuesValuesData = outValuesValues->template mutable_data<V>();

            int outKeysOffset = 0;
            int outValuesValuesOffset = 0;
            for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
              inKeysOffset_[inputIndex] = 0;
              inValuesValuesOffset_[inputIndex] = 0;
            }
            for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
              outLengthsData[exampleIndex] = 0;
              for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
                const int32_t* inLengthsData =
                    Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
                const int64_t* inKeysData = Input(kNumTensorsPerInput * inputIndex + 1)
                                                .template data<int64_t>();
                const int32_t* inValuesLengthsData =
                    Input(kNumTensorsPerInput * inputIndex + 2)
                        .template data<int32_t>();
                const auto& inValuesKeys = Input(kNumTensorsPerInput * inputIndex + 3);
                const auto& inValuesValues =
                    Input(kNumTensorsPerInput * inputIndex + 4);
                outLengthsData[exampleIndex] += inLengthsData[exampleIndex];
                for (int featureIndex = 0; featureIndex < inLengthsData[exampleIndex];
                     ++featureIndex) {
                  outKeysData[outKeysOffset] = inKeysData[inKeysOffset_[inputIndex]];
                  outValuesLengthsData[outKeysOffset] =
                      inValuesLengthsData[inKeysOffset_[inputIndex]];
                  context_.CopyItemsSameDevice(
                      inValuesKeys.dtype(),
                      inValuesLengthsData[inKeysOffset_[inputIndex]],
                      &inValuesKeys
                           .template data<K>()[inValuesValuesOffset_[inputIndex]],
                      &outValuesKeysData[outValuesValuesOffset]);
                  context_.CopyItemsSameDevice(
                      inValuesValues.dtype(),
                      inValuesLengthsData[inKeysOffset_[inputIndex]],
                      &inValuesValues
                           .template data<V>()[inValuesValuesOffset_[inputIndex]],
                      &outValuesValuesData[outValuesValuesOffset]);
                  outValuesValuesOffset +=
                      inValuesLengthsData[inKeysOffset_[inputIndex]];
                  inValuesValuesOffset_[inputIndex] +=
                      inValuesLengthsData[inKeysOffset_[inputIndex]];
                  ++outKeysOffset;
                  ++inKeysOffset_[inputIndex];
                }
              }
            }

            return true;
        */
    }
}

