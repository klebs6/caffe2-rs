crate::ix!();

pub struct GetQuantDecodeGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetQuantDecodeGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(Def().input_size(), Def().output_size() + 1);
        vector<string> gradient_op_inputs;
        for (int i = 0; i < Def().input_size(); i++) {
          gradient_op_inputs.push_back(I(i));
        }
        for (int i = 0; i < Def().output_size(); i++) {
          gradient_op_inputs.push_back(GO(i));
        }
        return SingleGradientDef(
            "QuantDecodeGradient", "", gradient_op_inputs, vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    QuantDecode, 
    GetQuantDecodeGradient
}
