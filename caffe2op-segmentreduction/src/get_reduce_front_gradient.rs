crate::ix!();

pub struct GetReduceFrontGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceFrontGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // Have utility function generating these names?
          string tmp_dims = "_" + O(0) + "_dims";

          vector<string> grad_ins;
          for (const int i : ReducerGradient::originalInputs()) {
            grad_ins.push_back(I(i));
          }
          grad_ins.push_back(GO(0));
          grad_ins.push_back(tmp_dims);

          vector<Argument> args;
          if (ArgumentHelper::HasArgument(def_, "num_reduce_dim")) {
            args.push_back(GetArgument(def_, "num_reduce_dim"));
          }
          // FIXME: pass in num_reduce_dims?!
          return vector<OperatorDef>{
              CreateOperatorDef(
                  "Shape", "", vector<string>{I(0)}, vector<string>{tmp_dims}),
              CreateOperatorDef(
                  string(basename) + ReducerDef::name + "Gradient",
                  "",
                  grad_ins,
                  // no gradient on auxiliary inputs for now
                  vector<string>{GI(0)}),
          };
        */
    }
}
