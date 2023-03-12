crate::ix!();

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
