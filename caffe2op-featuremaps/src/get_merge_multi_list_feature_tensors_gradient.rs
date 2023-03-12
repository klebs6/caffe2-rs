crate::ix!();

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
