crate::ix!();

/**
  | Some Casts are compatible with gradients,
  | but for now we don't support it
  | 
  | GRADIENT_NOT_IMPLEMENTED_YET(Cast);
  |
  */
pub struct GetCastGradient;

impl GetGradientDefs for GetCastGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<OperatorDef> defs = SingleGradientDef("Cast", "", vector<string>{GO(0)}, vector<string>{GI(0)});

        // now modify the arguments in defs[0]
        ArgumentHelper argsHelper(def_);

        auto to_name = cast::GetCastDataType(argsHelper, "to");

        CAFFE_ENFORCE(
            argsHelper.HasSingleArgumentOfType<string>("from_type") ||
                argsHelper.HasSingleArgumentOfType<int>("from_type"),
            "Argument 'from_type' of type int or string"
            " is required to get the gradient of CastOp");

        auto from_name = cast::GetCastDataType(argsHelper, "from_type");
        Argument *to = defs[0].add_arg();
        to->set_name("to");
        to->set_i(from_name);

        Argument *from = defs[0].add_arg();
        from->set_name("from_type");
        from->set_i(to_name);

        return defs;
        */
    }
}

impl CopyArguments for GetCastGradient {
    
    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_gradient!{Cast, GetCastGradient}
