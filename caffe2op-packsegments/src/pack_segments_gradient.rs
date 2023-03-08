crate::ix!();

pub struct GetPackSegmentsGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for  GetPackSegmentsGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {

        todo!();
        /*
           return SingleGradientDef(
           "UnpackSegments",
           "",
           vector<string>{I(0), GO(0)},
           vector<string>{GI(1)});
           */
    }
}

register_gradient!{
    PackSegments, 
    GetPackSegmentsGradient
}
