crate::ix!();

pub struct GetGRUUnitGradient;

impl GetGradientDefs for GetGRUUnitGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (GetFlagArgument(def_, "sequence_lengths", true)) {
              return SingleGradientDef(
                  "GRUUnitGradient",
                  "",
                  vector<string>{I(0), I(1), I(2), I(3), O(0), GO(0)},
                  vector<string>{GI(0), GI(1)});
            } else {
              return SingleGradientDef(
                  "GRUUnitGradient",
                  "",
                  vector<string>{I(0), I(1), I(2), O(0), GO(0)},
                  vector<string>{GI(0), GI(1)});
            }
        */
    }
}

register_gradient!{GRUUnit, GetGRUUnitGradient}
