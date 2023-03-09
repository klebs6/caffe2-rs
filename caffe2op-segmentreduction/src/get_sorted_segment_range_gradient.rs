crate::ix!();

pub struct GetSortedSegmentRangeGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSortedSegmentRangeGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
              "SortedSegmentRange" + ReducerDef::name + "Gradient",
              "",
              vector<string>{I(0), O(0), GO(0), I(1)},
              // no gradient on segment_ids!
              vector<string>{GI(0)});
        */
    }
}
