crate::ix!();

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
