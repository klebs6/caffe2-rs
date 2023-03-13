crate::ix!();

pub struct GetFlattenToVecGradient;

impl GetGradientDefs for GetFlattenToVecGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ResizeLike", "", vector<string>{GO(0), I(0)}, vector<string>{GI(0)});
        */
    }
}
