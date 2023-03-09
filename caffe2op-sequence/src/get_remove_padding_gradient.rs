crate::ix!();

pub struct GetRemovePaddingGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRemovePaddingGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // whether to provide lengths as input to gradient
        vector<std::string> g_inputs{GO(0)};
        if (Def().input_size() > 1) {
          CAFFE_ENFORCE(Def().output_size() > 1);
          g_inputs.push_back(O(1));
        }

        return SingleGradientDef("AddPadding", "", g_inputs, vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    RemovePadding, 
    GetRemovePaddingGradient
}
