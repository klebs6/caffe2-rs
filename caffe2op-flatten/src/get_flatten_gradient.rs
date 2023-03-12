crate::ix!();

pub struct GetFlattenGradient;

impl GetGradientDefs for GetFlattenGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ResizeLike", "", vector<string>{GO(0), I(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{Flatten, GetFlattenGradient}
