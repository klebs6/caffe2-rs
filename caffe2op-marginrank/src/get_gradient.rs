crate::ix!();

pub struct GetMarginRankingCriterionGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMarginRankingCriterionGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "MarginRankingCriterionGradient",
            "",
            vector<string>{I(0), I(1), I(2), GO(0)},
            vector<string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{
    MarginRankingCriterion, 
    GetMarginRankingCriterionGradient
}
