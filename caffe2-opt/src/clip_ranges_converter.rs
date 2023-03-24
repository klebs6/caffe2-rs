crate::ix!();

pub struct ClipRangesConverter { }

impl Converter for ClipRangesConverter {
    
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::ClipRanges>();
        const caffe2::ArgumentHelper args(op);
        auto c = dyn_cast<repr::ClipRanges>(nnOp.get());
        c->setMaxLength(args.GetSingleArgument<int64_t>("max_length", 0));
        return nnOp;
        */
    }
    
    #[inline] fn convert_to_operator_def(&mut self, nn_op: *const NeuralNetOperator) -> OperatorDef {
        
        todo!();
        /*
            auto clipRanges = dyn_cast<repr::ClipRanges>(nnOp);
        OperatorDef op;
        op.set_type("ClipRanges");
        op.add_arg()->CopyFrom(caffe2::MakeArgument<int64_t>(
            "max_length", clipRanges->getMaxLength()));
        op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
        return op;
        */
    }
}

register_converter!{ClipRanges, ClipRangesConverter}


