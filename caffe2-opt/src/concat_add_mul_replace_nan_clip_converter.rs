crate::ix!();

pub struct ConcatAddMulReplaceNaNClipConverter { }

impl Converter for ConcatAddMulReplaceNaNClipConverter {
    
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::ConcatAddMulReplaceNaNClip>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::ConcatAddMulReplaceNaNClip>(nnOp.get());
        if (argMap.count("clip_min")) {
          CAFFE_ENFORCE(argMap["clip_min"].has_f(), "Invalid 'clip_min' argument");
          c->setClipMin(static_cast<float>(argMap["clip_min"].f()));
        }
        if (argMap.count("clip_max")) {
          CAFFE_ENFORCE(argMap["clip_max"].has_f(), "Invalid 'clip_max' argument");
          c->setClipMin(static_cast<float>(argMap["clip_max"].f()));
        }
        return nnOp;
        */
    }
    
    #[inline] fn convert_to_operator_def(&mut self, nn_op: *const NeuralNetOperator) -> OperatorDef {
        
        todo!();
        /*
            auto cc_amrc = dyn_cast<repr::ConcatAddMulReplaceNaNClip>(nnOp);
        OperatorDef op;
        op.set_type("ConcatAddMulReplaceNaNClip");
        auto min_arg = op.add_arg();
        min_arg->set_name("clip_min");
        min_arg->set_f(cc_amrc->getClipMin());
        auto max_arg = op.add_arg();
        max_arg->set_name("clip_max");
        max_arg->set_f(cc_amrc->getClipMax());
        op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
        return op;
        */
    }
}

register_converter!{
    ConcatAddMulReplaceNaNClip,
    ConcatAddMulReplaceNaNClipConverter
}


