crate::ix!();

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
