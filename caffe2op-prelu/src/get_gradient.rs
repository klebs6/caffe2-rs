crate::ix!();

pub struct GetPReluGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetPReluGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            vector<string>{O(0), GO(0), I(0), I(1)},
            vector<string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{
    PRelu, 
    GetPReluGradient
}
