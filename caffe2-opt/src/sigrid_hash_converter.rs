crate::ix!();

pub struct SigridHashConverter { }

impl Converter for SigridHashConverter {
    
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::SigridHash>();
        const caffe2::ArgumentHelper args(op);
        auto c = dyn_cast<repr::SigridHash>(nnOp.get());
        c->setSalt(args.GetSingleArgument<int64_t>("salt", 0));
        c->setMaxValue(args.GetSingleArgument<int64_t>("maxValue", 0));
        c->setHashIntoInt32(args.GetSingleArgument<bool>("hashIntoInt32", false));
        return nnOp;
        */
    }
    
    #[inline] fn convert_to_operator_def(&mut self, nn_op: *const NeuralNetOperator) -> OperatorDef {
        
        todo!();
        /*
            auto sigridHash = dyn_cast<repr::SigridHash>(nnOp);
        OperatorDef op;
        op.set_type("SigridHash");
        op.add_arg()->CopyFrom(
            caffe2::MakeArgument<int64_t>("salt", sigridHash->getSalt()));
        op.add_arg()->CopyFrom(
            caffe2::MakeArgument<int64_t>("maxValue", sigridHash->getMaxValue()));
        op.add_arg()->CopyFrom(caffe2::MakeArgument<bool>(
            "hashIntoInt32", sigridHash->getHashIntoInt32()));
        op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
        return op;
        */
    }
}

register_converter!{SigridHash, SigridHashConverter}
