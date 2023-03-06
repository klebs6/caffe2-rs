crate::ix!();

pub struct GetLengthsTileGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLengthsTileGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(def_.input_size(), 2);
        return SingleGradientDef(
            "LengthsSum",
            "",
            // input 1 is the lengths used to repeat
            // DATA in the forward pass
            vector<string>{GO(0), I(1)},
            // only concerned with the gradient on "DATA"
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{LengthsTile, GetLengthsTileGradient}
