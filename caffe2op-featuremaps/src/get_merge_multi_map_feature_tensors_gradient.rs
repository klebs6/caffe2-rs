crate::ix!();

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
