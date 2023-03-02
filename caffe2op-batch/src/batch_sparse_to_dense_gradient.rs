crate::ix!();

pub struct GetBatchSparseToDenseGradient;

impl GetGradientDefs for GetBatchSparseToDenseGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BatchDenseToSparse",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(2)});
        */
    }
}

register_gradient!{BatchSparseToDense, GetBatchSparseToDenseGradient}
