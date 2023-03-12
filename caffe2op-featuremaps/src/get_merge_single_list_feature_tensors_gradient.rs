crate::ix!();

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
