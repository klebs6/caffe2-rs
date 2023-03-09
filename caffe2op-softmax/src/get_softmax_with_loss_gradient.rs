crate::ix!();

pub struct GetSoftmaxWithLossGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSoftmaxWithLossGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> blob_names{
            {I(0), I(1), O(0), GO(1)},
        };

        // Add weight blob, if given
        if (def_.input_size() == 3) {
          blob_names.emplace(blob_names.begin() + 2, I(2));
        }
        return SingleGradientDef(
            "SoftmaxWithLossGradient", "", blob_names, vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    SoftmaxWithLoss, 
    GetSoftmaxWithLossGradient
}
