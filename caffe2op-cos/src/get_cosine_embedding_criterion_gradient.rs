crate::ix!();

pub struct GetCosineEmbeddingCriterionGradient {

}

impl GetGradientDefs for GetCosineEmbeddingCriterionGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CosineEmbeddingCriterionGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}
