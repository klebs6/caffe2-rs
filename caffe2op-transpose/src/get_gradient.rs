crate::ix!();

pub struct GetTransposeGradient;

impl CopyArguments for GetTransposeGradient {

    /// We will create our own arguments.
    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

impl GetGradientDefs for GetTransposeGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            auto ops = SingleGradientDef(
            "Transpose", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        ops[0].mutable_arg()->CopyFrom(Def().arg());
        if (ArgumentHelper::HasArgument(Def(), "axes")) {
          // If axes is specified, we will need to figure out the inverse index.
          const Argument& old_axes = GetArgument(Def(), "axes");
          const int axes_size = old_axes.ints_size();
          Argument* new_arg = GetMutableArgument("axes", false, &ops[0]);
          for (int i = 0; i < axes_size; ++i) {
            new_arg->set_ints(old_axes.ints(i), i);
          }
        }
        return ops;
        */
    }
}

register_gradient!{Transpose, GetTransposeGradient}
