crate::ix!();

pub struct GetPoolGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetPoolGradient<'a> {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            vector<string>{I(0), O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{LpPool, GetPoolGradient}

