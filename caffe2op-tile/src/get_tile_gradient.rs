crate::ix!();

pub struct GetTileGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetTileGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // Check whether the tiles/axis information was
        // passed through input arguments
        std::vector<std::string> g_inputs({GO(0)});
        if (Def().input_size() > 1) {
          g_inputs.push_back(I(1));
        }
        if (Def().input_size() > 2) {
          g_inputs.push_back(I(2));
        }
        return SingleGradientDef(
            "TileGradient", "", g_inputs, std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Tile, GetTileGradient}
