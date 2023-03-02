crate::ix!();

pub struct GetClipGradient;

impl GetGradientDefs for GetClipGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ClipGradient", "",
            vector<string>{O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{Clip, GetClipGradient}

