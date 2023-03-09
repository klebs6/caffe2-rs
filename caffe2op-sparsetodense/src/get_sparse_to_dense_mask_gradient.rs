crate::ix!();

pub struct GetSparseToDenseMaskGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSparseToDenseMaskGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> blob_names{I(0), GO(0)};

        // Add lengths blob if given
        if (def_.input_size() == 4) {
          blob_names.push_back(I(3));
        }
        return SingleGradientDef(
            "SparseToDenseMaskGradient", "", blob_names, vector<string>{GI(1)});
        */
    }
}

register_gradient!{SparseToDenseMask, GetSparseToDenseMaskGradient}

export_caffe2_op_to_c10_cpu!{SparseToDenseMask,
    "_caffe2::SparseToDenseMask(
        Tensor indices, 
        Tensor values, 
        Tensor default_value, 
        Tensor? lengths, 
        int[] mask, 
        bool? return_presence_mask = False, 
        int? max_skipped_indices = 50) -> (
        Tensor output, 
        Tensor presence_mask)",
    SparseToDenseMaskOp<CPUContext>
}
