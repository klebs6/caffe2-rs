crate::ix!();

/**
  | Merge given single-feature tensors
  | with map features into one multi-feature
  | tensor.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeSingleMapFeatureTensorsOp<Context> {

    storage:                 OperatorStorage,
    context:                 Context,

    k_num_tensors_per_input: i32, // = 4;
    num_inputs:              i32,
    in_values_offset:        Vec<i32>,
    feature_ids:             Vec<i64>,
}

num_inputs!{MergeSingleMapFeatureTensors, 
    |n: i32| {
        n >= 4 && n % 4 == 0
    }
}

num_outputs!{MergeSingleMapFeatureTensors, 5}

inputs!{MergeSingleMapFeatureTensors, 
    0 => ("in1_lengths",         ".lengths"),
    1 => ("in1_keys",            ".keys"),
    2 => ("in1_values",          ".values"),
    3 => ("in1_presence",        ".presence")
}

outputs!{MergeSingleMapFeatureTensors, 
    0 => ("out_lengths",         ".lengths"),
    1 => ("out_keys",            ".keys"),
    2 => ("out_values_lengths",  ".values.lengths"),
    3 => ("out_values_keys",     ".values.keys"),
    4 => ("out_values_values",   ".values.values")
}

args!{MergeSingleMapFeatureTensors, 
    0 => ("feature_ids",         "feature ids")
}

impl<Context> MergeSingleMapFeatureTensorsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        numInputs_ = InputSize() / kNumTensorsPerInput;
        inValuesOffset_.resize(numInputs_);
        featureIDs_ = this->template GetRepeatedArgument<int64_t>("feature_ids");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<bool, int32_t, int64_t, float, double, std::string>>::
            call(this, Input(1));
        */
    }

    #[inline] pub fn do_run_with_type<K>(&mut self) -> bool {
        todo!();
        /*
            return DispatchHelper<
                TensorTypes2<bool, int32_t, int64_t, float, double, std::string>,
                K>::call(this, Input(2));
        */
    }

    #[inline] pub fn do_run_with_type2<K, V>(&mut self) -> bool {
        todo!();
        /*
            int numExamples = Input(0).numel();
            int totalNumFeatures = 0;
            int totalNumValues = 0;
            for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
              const int32_t* inLengthsData =
                  Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
              const bool* inPresenceData =
                  Input(kNumTensorsPerInput * inputIndex + 3).template data<bool>();
              for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
                if (inPresenceData[exampleIndex]) {
                  ++totalNumFeatures;
                  totalNumValues += inLengthsData[exampleIndex];
                }
              }
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

            int keysOffset = 0;
            int valuesOffset = 0;
            for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
              inValuesOffset_[inputIndex] = 0;
            }
            for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
              outLengthsData[exampleIndex] = 0;
              for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
                const int32_t* inLengthsData =
                    Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
                const auto& inKeys = Input(kNumTensorsPerInput * inputIndex + 1);
                const auto& inValues = Input(kNumTensorsPerInput * inputIndex + 2);
                const bool* inPresenceData =
                    Input(kNumTensorsPerInput * inputIndex + 3).template data<bool>();
                if (inPresenceData[exampleIndex]) {
                  ++outLengthsData[exampleIndex];
                  outKeysData[keysOffset] = featureIDs_[inputIndex];
                  outValuesLengthsData[keysOffset] = inLengthsData[exampleIndex];
                  context_.CopyItemsSameDevice(
                      inKeys.dtype(),
                      inLengthsData[exampleIndex],
                      &inKeys.template data<K>()[inValuesOffset_[inputIndex]],
                      &outValuesKeysData[valuesOffset]);
                  context_.CopyItemsSameDevice(
                      inValues.dtype(),
                      inLengthsData[exampleIndex],
                      &inValues.template data<V>()[inValuesOffset_[inputIndex]],
                      &outValuesValuesData[valuesOffset]);
                  valuesOffset += inLengthsData[exampleIndex];
                  inValuesOffset_[inputIndex] += inLengthsData[exampleIndex];
                  ++keysOffset;
                }
              }
            }
            return true;
        */
    }
}

