crate::ix!();

pub struct GetLeakyReluGradient;

impl GetGradientDefs for GetLeakyReluGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "LeakyReluGradient",
            "",
            vector<string>{O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{LeakyRelu, GetLeakyReluGradient}
