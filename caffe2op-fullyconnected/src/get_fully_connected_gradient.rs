crate::ix!();

pub struct GetFCGradient;

impl GetGradientDefs for GetFCGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(def_.input_size(), 3);
        CAFFE_ENFORCE(def_.type() == "FC" || def_.type() == "FCTransposed");
        return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(1), GI(2), GI(0)});
        */
    }
}

register_gradient!{FC, GetFCGradient}

register_gradient!{FCTransposed, GetFCGradient}
