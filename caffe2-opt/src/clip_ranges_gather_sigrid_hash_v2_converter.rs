crate::ix!();

pub struct ClipRangesGatherSigridHashV2Converter { }

impl Converter for ClipRangesGatherSigridHashV2Converter { 
    
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::ClipRangesGatherSigridHashV2>();
        const caffe2::ArgumentHelper args(op);

        auto c = dyn_cast<repr::ClipRangesGatherSigridHashV2>(nnOp.get());
        if (args.HasArgument("max_lengths")) {
          c->setMaxLengths(args.GetRepeatedArgument<int64_t>("max_lengths"));
        }
        if (args.HasArgument("salts")) {
          c->setSalts(args.GetRepeatedArgument<int64_t>("salts"));
        }
        if (args.HasArgument("max_values")) {
          c->setMaxValues(args.GetRepeatedArgument<int64_t>("max_values"));
        }
        if (args.HasArgument("hash_into_int32")) {
          c->setHashIntoInt32(
              args.GetSingleArgument<bool>("hash_into_int32", false));
        }
        return nnOp;
        */
    }
    
    #[inline] fn convert_to_operator_def(&mut self, nn_op: *const NeuralNetOperator) -> OperatorDef {
        
        todo!();
        /*
            auto fuse = dyn_cast<repr::ClipRangesGatherSigridHashV2>(nnOp);
        OperatorDef op;
        op.set_type("ClipRangesGatherSigridHashV2");
        op.add_arg()->CopyFrom(caffe2::MakeArgument<vector<int64_t>>(
            "max_lengths", fuse->getMaxLengths()));
        op.add_arg()->CopyFrom(
            caffe2::MakeArgument<vector<int64_t>>("salts", fuse->getSalts()));
        op.add_arg()->CopyFrom(caffe2::MakeArgument<vector<int64_t>>(
            "max_values", fuse->getMaxValues()));
        op.add_arg()->CopyFrom(caffe2::MakeArgument<bool>(
            "hash_into_int32", fuse->getHashIntoInt32()));
        op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
        return op;
        */
    }
}

register_converter!{
    ClipRangesGatherSigridHashV2,
    ClipRangesGatherSigridHashV2Converter
}

