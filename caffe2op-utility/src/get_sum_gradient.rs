crate::ix!();

pub struct GetSumGradient;

impl GetGradientDefs for GetSumGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            for (auto i = 0; i < def_.input_size(); ++i) {
          SetDense(i, GO(0));
        }
        return vector<OperatorDef>();
        */
    }
}

register_gradient!{Sum, GetSumGradient}
