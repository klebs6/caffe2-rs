crate::ix!();

pub trait ConvToNNPackTransform : SingleOpTransform {

    /**
      | Specify what the op needs to be to match
      | the pattern.
      |
      */
    #[inline] fn match_operator(&mut self, op: &OperatorDef) -> bool {
        
        todo!();
        /*
            return (
            op.type() == "Conv" && op.device_option().device_type() == PROTO_CPU &&
            op.engine() != "NNPACK");
        */
    }

    /**
      | Specify how the operator should be replaced.
      |
      */
    #[inline] fn replace_operator(&mut self, op: *mut OperatorDef)  {
        
        todo!();
        /*
            op->set_engine("NNPACK");
        */
    }
}

register_transform!{ConvToNNPack, ConvToNNPackTransform}
