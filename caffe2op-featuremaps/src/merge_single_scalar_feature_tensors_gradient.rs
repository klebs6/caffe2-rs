crate::ix!();

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
