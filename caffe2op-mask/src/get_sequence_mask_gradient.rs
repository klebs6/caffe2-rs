crate::ix!();

pub struct GetSequenceMaskGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSequenceMaskGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<Argument> args;
        args.reserve(Def().arg().size());
        for (const auto& x : Def().arg()) {
          args.push_back(x);
        }
        args.push_back(MakeArgument<bool>("grad", true));
        if (def_.input_size() == 1) {
          return SingleGradientDef(
              "SequenceMask",
              "",
              vector<string>{GO(0)},
              vector<string>{GI(0)},
              args);
        } else {
          return SingleGradientDef(
              "SequenceMask",
              "",
              vector<string>{GO(0), I(1)},
              vector<string>{GI(0)},
              args);
        }
        */
    }
}

impl<'a> CopyArguments for GetSequenceMaskGradient<'a> {

    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_gradient!{SequenceMask, GetSequenceMaskGradient}
