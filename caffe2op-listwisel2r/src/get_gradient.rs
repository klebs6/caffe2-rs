crate::ix!();

pub struct GetLambdaRankNdcgGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLambdaRankNdcgGradient<'a> {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "LambdaRankNdcgGradient",
            "",
            vector<string>{I(0), I(2), O(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{LambdaRankNdcg, GetLambdaRankNdcgGradient}


