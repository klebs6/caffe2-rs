crate::ix!();

pub struct GetUnpackSegmentsGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetUnpackSegmentsGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "PackSegments", "", vector<string>{I(0), GO(0)}, vector<string>{GI(1)});
        */
    }
}

register_gradient!{
    UnpackSegments, 
    GetUnpackSegmentsGradient
}
