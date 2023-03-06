crate::ix!();

pub struct GetBooleanMaskGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetBooleanMaskGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BooleanMaskGradient",
            "",
            vector<string>{I(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{BooleanMask, GetBooleanMaskGradient}
