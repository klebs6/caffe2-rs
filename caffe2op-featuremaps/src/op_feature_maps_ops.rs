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

use crate::{
    OperatorStorage,
    GradientMakerBase,
    OperatorDef,
};

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

/**
  | Explode multi-feature tensor of scalar
  | features into one or more single-feature
  | tensors
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeSingleScalarFeatureTensorsGradientOp<Context> {

    storage:            OperatorStorage,
    context:            Context,
    num_feature_inputs: i32,
}

num_inputs!{MergeSingleScalarFeatureTensorsGradient, 
    |n: i32| {
        n >= 2
    }
}

num_outputs!{MergeSingleScalarFeatureTensorsGradient, 
    |n: i32| {
        n >= 1
    }
}

inputs!{MergeSingleScalarFeatureTensorsGradient, 
    0 => ("in1_presence", ".presence"),
    1 => (".values_grad", ".values_grad")
}

outputs!{MergeSingleScalarFeatureTensorsGradient, 
    0 => ("in1_grad",     "_grad of inputs")
}

impl<Context> MergeSingleScalarFeatureTensorsGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        numFeatureInputs_ = InputSize() - 1; // Everything other than values_grad
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
            for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
              Output(inputIndex)->ResizeLike(Input(inputIndex));
            }

            const T* inValuesGradData = Input(InputSize() - 1).template data<T>();

            T default_value = T();
            int valuesOffset = 0;
            for (int exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
              for (int inputIndex = 0; inputIndex < numFeatureInputs_; ++inputIndex) {
                const bool* inPresenceData = Input(inputIndex).template data<bool>();
                T* outFeatureData = Output(inputIndex)->template mutable_data<T>();
                if (inPresenceData[exampleIndex]) {
                  outFeatureData[exampleIndex] = inValuesGradData[valuesOffset];
                  ++valuesOffset;
                } else {
                  outFeatureData[exampleIndex] = default_value;
                }
              }
            }
            return true;
        */
    }
}

pub struct GetMergeSingleScalarFeatureTensorsGradient;

impl GetGradientDefs for GetMergeSingleScalarFeatureTensorsGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> input_blob_names{};
        vector<string> output_blob_names{};

        for (int inputIdx = 0; inputIdx < def_.input_size() / 2; ++inputIdx) {
          input_blob_names.push_back(I(inputIdx * 2 + 1));
          output_blob_names.push_back(GI(inputIdx * 2));
        }
        input_blob_names.push_back(GO(2));

        return SingleGradientDef(
            "MergeSingleScalarFeatureTensorsGradient",
            "", /* name */
            input_blob_names,
            output_blob_names);
        */
    }
}

/**
  | Merge given single-feature tensors
  | with list features into one multi-feature
  | tensor.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeSingleListFeatureTensorsOp<Context> {

    storage:                 OperatorStorage,
    context:                 Context,

    k_num_tensors_per_input: i32, // = 3;
    num_inputs:              i32,
    in_values_offset:        Vec<i32>,
    feature_ids:             Vec<i64>,
}

num_inputs!{MergeSingleListFeatureTensors, 
    |n: i32| {
        n >= 3 && n % 3 == 0
    }
}
    
num_outputs!{MergeSingleListFeatureTensors, 4}

inputs!{MergeSingleListFeatureTensors, 
    0 => ("in1_lengths",         ".lengths"),
    1 => ("in1_values",          ".values"),
    2 => ("in1_presence",        ".presence")
}

outputs!{MergeSingleListFeatureTensors, 
    0 => ("out_lengths",         ".lengths"),
    1 => ("out_keys",            ".keys"),
    2 => ("out_values_lengths",  ".values.lengths"),
    3 => ("out_values_values",   ".values.values")
}

args!{MergeSingleListFeatureTensors, 
    0 => ("feature_ids",         "feature ids")
}

impl<Context> MergeSingleListFeatureTensorsOp<Context> {
    
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

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            int numExamples = Input(0).numel();
            int totalNumFeatures = 0;
            int totalNumValues = 0;
            for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
              const int32_t* inLengthsData =
                  Input(kNumTensorsPerInput * inputIndex).template data<int32_t>();
              const bool* inPresenceData =
                  Input(kNumTensorsPerInput * inputIndex + 2).template data<bool>();
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
            auto* outValuesValues = Output(3, {totalNumValues}, at::dtype<T>());

            int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
            int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
            int32_t* outValuesLengthsData =
                outValuesLengths->template mutable_data<int32_t>();
            T* outValuesValuesData = outValuesValues->template mutable_data<T>();

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
                const auto& inValues = Input(kNumTensorsPerInput * inputIndex + 1);
                const bool* inPresenceData =
                    Input(kNumTensorsPerInput * inputIndex + 2).template data<bool>();
                if (inPresenceData[exampleIndex]) {
                  ++outLengthsData[exampleIndex];
                  outKeysData[keysOffset] = featureIDs_[inputIndex];
                  outValuesLengthsData[keysOffset] = inLengthsData[exampleIndex];
                  context_.CopyItemsSameDevice(
                      inValues.dtype(),
                      inLengthsData[exampleIndex],
                      &inValues.template data<T>()[inValuesOffset_[inputIndex]],
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

/**
  | Explode multi-feature tensors with
  | list features into single-feature
  | tensors.
  |
  */
pub struct GetMergeSingleListFeatureTensorsGradient;

impl GetGradientDefs for GetMergeSingleListFeatureTensorsGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> input_blob_names{};
        vector<string> output_blob_names{};

        for (int inputIdx = 0; inputIdx < def_.input_size() / 3; ++inputIdx) {
          input_blob_names.push_back(I(inputIdx * 3));
          input_blob_names.push_back(I(inputIdx * 3 + 2));
          output_blob_names.push_back(GI(inputIdx * 3 + 1));
        }
        input_blob_names.push_back(GO(3));

        return SingleGradientDef(
            "MergeSingleListFeatureTensorsGradient",
            "",
            input_blob_names,
            output_blob_names);
        */
    }
}

num_inputs!{MergeSingleListFeatureTensorsGradient, 
    |n: i32| {
        n >= 3 && n % 2 == 1
    }
}

num_outputs!{MergeSingleListFeatureTensorsGradient, 
    |n: i32| {
        n >= 1
    }
}

inputs!{MergeSingleListFeatureTensorsGradient, 
    0 => ("in1_lengths",        ".lengths"),
    1 => ("in1_presence",       ".presence"),
    2 => ("out_values_values",  ".values.values_grad")
}

outputs!{MergeSingleListFeatureTensorsGradient, 
    0 => ("out1_values",        ".values_grad")
}

///-------------------------------------------
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

/**
  | Explode given multi-feature tensors
  | with map features into multiple single-feature
  | tensor.
  |
  */
pub struct GetMergeSingleMapFeatureTensorsGradient;

num_inputs!{MergeSingleMapFeatureTensorsGradient, 
    |n: i32| {
        n >= 3 && n % 2 == 1
    }
}

num_outputs!{MergeSingleMapFeatureTensorsGradient, 
    |n: i32| {
        n >= 1
    }
}

inputs!{MergeSingleMapFeatureTensorsGradient, 
    0 => ("in1_lengths",            ".lengths"),
    1 => ("in1_presence",           ".presence"),
    2 => ("out_values_values_grad", ".values.values_grad")
}

outputs!{MergeSingleMapFeatureTensorsGradient, 
    0 => ("in1_values_grad",        ".values_grad")
}

impl GetGradientDefs for GetMergeSingleMapFeatureTensorsGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> input_blob_names{};
        vector<string> output_blob_names{};

        for (int inputIdx = 0; inputIdx < def_.input_size() / 4; ++inputIdx) {
          input_blob_names.push_back(I(inputIdx * 4));
          input_blob_names.push_back(I(inputIdx * 4 + 3));
          output_blob_names.push_back(GI(inputIdx * 4 + 2));
        }
        input_blob_names.push_back(GO(4));

        return SingleGradientDef(
            "MergeSingleMapFeatureTensorsGradient",
            "",
            input_blob_names,
            output_blob_names);
        */
    }
}

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

pub struct GetMergeMultiScalarFeatureTensorsGradient {
    num_tensors_per_input: i32,// = 3;
}

impl GetGradientDefs for GetMergeMultiScalarFeatureTensorsGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> input_blob_names{};
        vector<string> output_blob_names{};

        for (int inputIdx = 0; inputIdx < def_.input_size() / kNumTensorsPerInput;
             ++inputIdx) {
          input_blob_names.push_back(I(inputIdx * kNumTensorsPerInput));
          output_blob_names.push_back(GI(inputIdx * kNumTensorsPerInput + 2));
        }
        input_blob_names.push_back(GO(2));

        return SingleGradientDef(
            "MergeMultiScalarFeatureTensorsGradient",
            "",
            input_blob_names,
            output_blob_names);
        */
    }
}

/**
  | Merge given multi-feature tensors
  | with list features into one.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeMultiListFeatureTensorsOp<Context> {

    storage:                 OperatorStorage,
    context:                 Context,
    k_num_tensors_per_input: i32, // = 4;
    num_inputs:              i32,
    in_keys_offset:          Vec<i32>,
    in_values_values_offset: Vec<i32>,
}

num_outputs!{MergeMultiListFeatureTensors, 4}

num_inputs!{MergeMultiListFeatureTensors, 
    |n: i32| {
        n >= 4 && n % 4 == 0
    }
}

inputs!{MergeMultiListFeatureTensors, 
    0 => ("in1_lengths",        ".lengths"),
    1 => ("in1_keys",           ".keys"),
    2 => ("in1_values_lengths", ".values.lengths"),
    3 => ("in1_values_values",  ".values.values")
}

outputs!{MergeMultiListFeatureTensors, 
    0 => ("out_lengths",        ".lengths"),
    1 => ("out_keys",           ".keys"),
    2 => ("out_values_lengths", ".values.lengths"),
    3 => ("out_values_values",  ".values.values")
}

impl<Context> MergeMultiListFeatureTensorsOp<Context> {
    
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

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            int numExamples = Input(0).numel();
            int totalNumFeatures = 0;
            int totalNumValues = 0;
            for (int inputIndex = 0; inputIndex < numInputs_; ++inputIndex) {
              totalNumFeatures += Input(kNumTensorsPerInput * inputIndex + 1).numel();
              totalNumValues += Input(kNumTensorsPerInput * inputIndex + 3).numel();
            }

            auto* outLengths = Output(0, {numExamples}, at::dtype<int32_t>());
            auto* outKeys = Output(1, {totalNumFeatures}, at::dtype<int64_t>());
            auto* outValuesLengths =
                Output(2, {totalNumFeatures}, at::dtype<int32_t>());
            auto* outValuesValues = Output(3, {totalNumValues}, at::dtype<T>());

            int32_t* outLengthsData = outLengths->template mutable_data<int32_t>();
            int64_t* outKeysData = outKeys->template mutable_data<int64_t>();
            int32_t* outValuesLengthsData =
                outValuesLengths->template mutable_data<int32_t>();
            T* outValuesValuesData = outValuesValues->template mutable_data<T>();

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
                const auto& inValuesValues =
                    Input(kNumTensorsPerInput * inputIndex + 3);
                outLengthsData[exampleIndex] += inLengthsData[exampleIndex];
                for (int featureIndex = 0; featureIndex < inLengthsData[exampleIndex];
                     ++featureIndex) {
                  outKeysData[outKeysOffset] = inKeysData[inKeysOffset_[inputIndex]];
                  outValuesLengthsData[outKeysOffset] =
                      inValuesLengthsData[inKeysOffset_[inputIndex]];
                  context_.CopyItemsSameDevice(
                      inValuesValues.dtype(),
                      inValuesLengthsData[inKeysOffset_[inputIndex]],
                      &inValuesValues
                           .template data<T>()[inValuesValuesOffset_[inputIndex]],
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

/**
  | Explode given multi-feature tensors
  | with list features into many.
  |
  */
pub struct GetMergeMultiListFeatureTensorsGradient {
    num_tensors_per_input: i32,// = 4;
}

inputs!{MergeMultiListFeatureTensorsGradient, 
    0 => ("in1_lengths",            ".lengths"),
    1 => ("in1_values_lengths",     ".values.lengths"),
    2 => ("out_values_values_grad", ".values.values_grad")
}

outputs!{MergeMultiListFeatureTensorsGradient, 
    0 => ("in1_values_values_grad", ".values.values_grad")
}

num_inputs!{MergeMultiListFeatureTensorsGradient, 
    |n: i32| {
        n >= 3 && n % 2 == 1
    }
}

num_outputs!{MergeMultiListFeatureTensorsGradient, 
    |n: i32| {
        n >= 1
    }
}

impl GetGradientDefs for GetMergeMultiListFeatureTensorsGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> input_blob_names{};
        vector<string> output_blob_names{};

        for (int inputIdx = 0; inputIdx < def_.input_size() / kNumTensorsPerInput;
             ++inputIdx) {
          input_blob_names.push_back(I(inputIdx * kNumTensorsPerInput));
          input_blob_names.push_back(I(inputIdx * kNumTensorsPerInput + 2));
          output_blob_names.push_back(GI(inputIdx * kNumTensorsPerInput + 3));
        }
        input_blob_names.push_back(GO(3));

        return SingleGradientDef(
            "MergeMultiListFeatureTensorsGradient",
            "",
            input_blob_names,
            output_blob_names);
        */
    }
}

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

/**
  | Explode given multi-feature tensors
  | with map features into many.
  |
  */
pub struct GetMergeMultiMapFeatureTensorsGradient {
    num_tensors_per_input: i32,// = 5;
}

impl GetGradientDefs for GetMergeMultiMapFeatureTensorsGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> input_blob_names{};
        vector<string> output_blob_names{};

        for (int inputIdx = 0; inputIdx < def_.input_size() / kNumTensorsPerInput;
             ++inputIdx) {
          input_blob_names.push_back(I(inputIdx * kNumTensorsPerInput));
          input_blob_names.push_back(I(inputIdx * kNumTensorsPerInput + 2));
          output_blob_names.push_back(GI(inputIdx * kNumTensorsPerInput + 4));
        }
        input_blob_names.push_back(GO(4));

        return SingleGradientDef(
            "MergeMultiMapFeatureTensorsGradient",
            "",
            input_blob_names,
            output_blob_names);
        */
    }
}

num_inputs!{MergeMultiMapFeatureTensorsGradient, 
    |n: i32| {
        n >= 3 && n % 2 == 1
    }
}

num_outputs!{MergeMultiMapFeatureTensorsGradient, 
    |n: i32| {
        n >= 1
    }
}

inputs!{MergeMultiMapFeatureTensorsGradient, 
    0 => ("in1_lengths",             ".lengths"),
    1 => ("in1_values_lengths",      ".values.lengths"),
    2 => ("out_values_values_grad",  ".values.values_grad")
}

outputs!{MergeMultiMapFeatureTensorsGradient, 
    0 => ("in1_values_values_grad",  ".values.values_grad")
}

///------------------------
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

register_cpu_operator!{MergeMultiListFeatureTensorsGradient,       MergeMultiListOrMapFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeMultiMapFeatureTensors,                MergeMultiMapFeatureTensorsOp<CPUContext>}
register_cpu_operator!{MergeMultiMapFeatureTensorsGradient,        MergeMultiListOrMapFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeMultiScalarFeatureTensors,             MergeMultiScalarFeatureTensorsOp<CPUContext>}
register_cpu_operator!{MergeMultiScalarFeatureTensorsGradient,     MergeMultiScalarFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeSingleListFeatureTensors,              MergeSingleListFeatureTensorsOp<CPUContext>}
register_cpu_operator!{MergeSingleListFeatureTensorsGradient,      MergeSingleListOrMapFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeSingleMapFeatureTensors,               MergeSingleMapFeatureTensorsOp<CPUContext>}
register_cpu_operator!{MergeSingleMapFeatureTensorsGradient,       MergeSingleListOrMapFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeSingleScalarFeatureTensorsGradient,    MergeSingleScalarFeatureTensorsGradientOp<CPUContext>}
register_cpu_operator!{MergeDenseFeatureTensors,                   MergeDenseFeatureTensorsOp<CPUContext>}
register_cpu_operator!{MergeSingleScalarFeatureTensors,            MergeSingleScalarFeatureTensorsOp<CPUContext>}

register_gradient!{MergeMultiListFeatureTensors,                   GetMergeMultiListFeatureTensorsGradient}
register_gradient!{MergeMultiMapFeatureTensors,                    GetMergeMultiMapFeatureTensorsGradient}
register_gradient!{MergeMultiScalarFeatureTensors,                 GetMergeMultiScalarFeatureTensorsGradient}
register_gradient!{MergeSingleListFeatureTensors,                  GetMergeSingleListFeatureTensorsGradient}
register_gradient!{MergeSingleMapFeatureTensors,                   GetMergeSingleMapFeatureTensorsGradient}
register_gradient!{MergeSingleScalarFeatureTensors,                GetMergeSingleScalarFeatureTensorsGradient}
