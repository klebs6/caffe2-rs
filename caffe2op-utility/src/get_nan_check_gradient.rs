crate::ix!();

pub struct GetNanCheckGradient;

impl GetGradientDefs for GetNanCheckGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return {CreateOperatorDef(
            "NanCheck",
            "",
            std::vector<string>{GO(0)},
            std::vector<string>{GI(0)})};
        */
    }
}
