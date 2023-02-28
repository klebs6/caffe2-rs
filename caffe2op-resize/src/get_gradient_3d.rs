crate::ix!();

pub struct GetResizeNearest3DGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetResizeNearest3DGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ResizeNearest3DGradient",
            "",
            vector<string>{GO(0), I(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{ResizeNearest3D, GetResizeNearest3DGradient}
