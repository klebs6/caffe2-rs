crate::ix!();

pub struct GetSoftsignGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSoftsignGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            I(0) != O(0),
            "Cannot compute softsign gradient "
            "if you choose to do an in-place calculation.");

        return SingleGradientDef(
            "SoftsignGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Softsign, GetSoftsignGradient}
