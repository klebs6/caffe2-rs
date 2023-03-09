crate::ix!();

/**
  | Spatial batch normalization's gradient,
  | depending on the various input sizes,
  | is a bit more complex than usual gradient
  | operators.
  |
  */
pub struct GetSpatialBNGradient { }

impl GetGradientDefs for GetSpatialBNGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // Check if we are in training or testing mode.
        const bool is_test =
            ArgumentHelper::GetSingleArgument(def_, OpSchema::Arg_IsTest, 0);
        const int num_batches =
            ArgumentHelper::GetSingleArgument(def_, "num_batches", 1);
        const std::vector<string> grad_outputs = {GI(0), GI(1), GI(2)};
        std::vector<string> grad_inputs;
        if (is_test) {
          // This is in testing mode. The operator should have five inputs:
          //     X, scale, bias, estimated_mean, estimated_variance
          // The gradient inputs are:
          //     X, scale, dY, estimated_mean, estimated_variance
          CAFFE_ENFORCE_EQ(def_.input_size(), 5);
          CAFFE_ENFORCE_EQ(def_.output_size(), 1);
          grad_inputs = std::vector<std::string>{I(0), I(1), GO(0), I(3), I(4)};
        } else if (num_batches > 1) {
          CAFFE_ENFORCE_EQ(def_.input_size(), 7);
          CAFFE_ENFORCE_EQ(def_.output_size(), 5);
          grad_inputs =
              std::vector<std::string>{I(0), I(1), GO(0), O(3), O(4), GI(1), GI(2)};
        } else {
          CAFFE_ENFORCE_EQ(def_.input_size(), 5);
          CAFFE_ENFORCE_EQ(def_.output_size(), 5);
          grad_inputs = std::vector<std::string>{I(0), I(1), GO(0), O(3), O(4)};
        }
        return SingleGradientDef(
            "SpatialBNGradient", "", grad_inputs, grad_outputs);
        */
    }
}
