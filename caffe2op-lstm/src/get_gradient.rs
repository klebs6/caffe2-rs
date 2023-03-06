crate::ix!();

pub struct GetLSTMUnitGradient;

impl GetGradientDefs for GetLSTMUnitGradient {
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (GetFlagArgument(def_, "sequence_lengths", true)) {
          return SingleGradientDef(
              "LSTMUnitGradient",
              "",
              vector<string>{
                  I(0), I(1), I(2), I(3), I(4), O(0), O(1), GO(0), GO(1)},
              vector<string>{GI(0), GI(1), GI(2)});
        } else {
          return SingleGradientDef(
              "LSTMUnitGradient",
              "",
              vector<string>{I(0), I(1), I(2), I(3), O(0), O(1), GO(0), GO(1)},
              vector<string>{GI(0), GI(1), GI(2)});
        }
        */
    }
}

register_gradient!{LSTMUnit, GetLSTMUnitGradient}
