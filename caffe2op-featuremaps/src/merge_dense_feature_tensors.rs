/*!
  | Single-feature representation:
  | - scalar features:
  |   <feature full name> T
  | - list features:
  |   <feature full name>.lengths int32
  |   <feature full name>.values T
  | - map features:
  |   <feature full name>.lengths int32
  |   <feature full name>.keys K
  |   <feature full name>.values V
  |
  | Missing values are set to zero, and value
  | presence flag is set accordingly:
  |
  |   <feature full name>.presence bool
  |
  | Multi-feature representation:
  | - scalar features:
  |   <feature type>.lengths int32
  |   <feature type>.keys int64
  |   <feature type>.values T
  | - list features:
  |   <feature type>.lengths int32
  |   <feature type>.keys int64
  |   <feature type>.values.lengths int32
  |   <feature type>.values.values T
  | - map features:
  |   <feature type>.lengths int32
  |   <feature type>.keys int64
  |   <feature type>.values.lengths int32
  |   <feature type>.values.keys K
  |   <feature type>.values.values V
  |
  | You can read more about representing batches of
  | lists and maps here:
  |
  | https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
  */

crate::ix!();

/**
  | Merge given multi-feature dense tensors
  | into one multi-feature tensor.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeDenseFeatureTensorsOp<Context> {
    storage:     OperatorStorage,
    context:     Context,
    feature_ids: Vec<i64>,
}

num_inputs!{MergeDenseFeatureTensors, 2}

num_outputs!{MergeDenseFeatureTensors, 3}

inputs!{MergeDenseFeatureTensors, 
    0 => ("in1",          ""),
    1 => ("in1_presence", ".presence")
}

outputs!{MergeDenseFeatureTensors, 
    0 => ("out_lengths", ".lengths"),
    1 => ("out_keys",    ".keys"),
    2 => ("out_values",  ".values")
}

args!{MergeDenseFeatureTensors, 
    0 => ("feature_ids", "feature ids")
}

impl<Context> MergeDenseFeatureTensorsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

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
            auto& dense_data = Input(0);
            int numExamples = dense_data.size(0);
            int numFeatures = dense_data.size(1);

            const bool* inPresenceData = Input(1).template data<bool>();
            int totalNumFeatures = 0;
            for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
              for (int inputIndex = 0; inputIndex < numFeatures; ++inputIndex) {
                if (inPresenceData[exampleIndex * numFeatures + inputIndex]) {
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
            const T* inData =
              Input(0).template data<T>();

            int keysOffset = 0;
            for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
              outLengthsData[exampleIndex] = 0;
              auto offset = exampleIndex * numFeatures;
              for (int inputIndex = 0; inputIndex < numFeatures; ++inputIndex) {
                if (inPresenceData[offset]) {
                  ++outLengthsData[exampleIndex];
                  outKeysData[keysOffset] = featureIDs_[inputIndex];
                  outValuesData[keysOffset] = inData[offset];
                  ++keysOffset;
                }
                offset++;
              }
            }
            return true;
        */
    }
}
