crate::ix!();

pub struct GetZeroGradientOpGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetZeroGradientOpGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
        return SingleGradientDef(
            "ConstantFill",
            "",
            vector<string>{I(0)},
            vector<string>{GI(0)},
            vector<Argument>{MakeArgument<float>("value", 0.0)});
        */
    }
}

register_gradient!{
    ZeroGradient, 
    GetZeroGradientOpGradient
}
