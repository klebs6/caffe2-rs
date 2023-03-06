crate::ix!();

pub struct GetAveragedLossGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetAveragedLossGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "AveragedLossGradient", "",
            vector<string>{I(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{AveragedLoss, GetAveragedLossGradient}
