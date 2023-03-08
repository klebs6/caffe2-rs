crate::ix!();

pub struct GetPairWiseLossGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetPairWiseLossGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> blob_names{I(0), I(1), GO(0)};

        // Add lengths blob if given
        if (def_.input_size() == 3) {
          blob_names.push_back(I(2));
        }
        return SingleGradientDef(
            "PairWiseLossGradient", "", blob_names, vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    PairWiseLoss, 
    GetPairWiseLossGradient
}
