crate::ix!();

pub struct GetPackRNNSequenceGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetPackRNNSequenceGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(def_.input_size(), 2);
        return SingleGradientDef(
            "UnpackRNNSequence",
            "",
            vector<string>{GO(0), I(1)},
            vector<string>{GI(0)});
        */
    }
}

pub struct GetUnpackRNNSequenceGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetUnpackRNNSequenceGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(def_.input_size(), 2);
        return SingleGradientDef(
            "PackRNNSequence",
            "",
            vector<string>{GO(0), I(1)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    PackRNNSequence,
    GetPackRNNSequenceGradient
}

register_gradient!{
    UnpackRNNSequence, 
    GetUnpackRNNSequenceGradient
}
