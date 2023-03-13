crate::ix!();


pub struct BatchMatMulConverter {
    base: dyn Converter,
}

impl BatchMatMulConverter {

    /**
      | Does not override default converter
      | to OperatorDef
      |
      */
    #[inline] pub fn convert_to_neural_net_operator(&mut self,
        op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::BatchMatMul>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::BatchMatMul>(nnOp.get());
        if (argMap.count("trans_a")) {
          CAFFE_ENFORCE(argMap["trans_a"].has_i(), "Invalid axis argument");
          int trans_a = static_cast<int>(argMap["trans_a"].i());
          c->setTransA(!!trans_a);
        }
        if (argMap.count("trans_b")) {
          CAFFE_ENFORCE(argMap["trans_b"].has_i(), "Invalid add_axis argument");
          int trans_b = static_cast<int>(argMap["trans_b"].i());
          c->setTransB(!!trans_b);
        }
        if (argMap.count("broadcast")) {
          CAFFE_ENFORCE(argMap["broadcast"].has_i(), "Invalid add_axis argument");
          int broadcast = static_cast<int>(argMap["broadcast"].i());
          c->setBroadcast(!!broadcast);
        }
        return nnOp;
        */
    }
}

register_converter!{BatchMatMul, BatchMatMulConverter}

trivial_converter!{BatchGather}

register_converter!{BatchGather, BatchGatherConverter}

pub struct MulConverter {
    base: dyn Converter,
}

impl MulConverter {
    
    /**
      | Does not override default converter
      | to OperatorDef
      |
      */
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::Mul>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::Mul>(nnOp.get());
        if (argMap.count("broadcast")) {
          CAFFE_ENFORCE(argMap["broadcast"].has_i(), "Invalid broadcast argument");
          int broadcast = static_cast<int>(argMap["broadcast"].i());
          c->setBroadcast(!!broadcast);
        }
        return nnOp;
        */
    }
}

register_converter!{Mul, MulConverter}

pub struct AddConverter { }

impl Converter for AddConverter {
    
    /**
      | Does not override default converter
      | to OperatorDef
      |
      */
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::Add>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::Add>(nnOp.get());
        if (argMap.count("broadcast")) {
          CAFFE_ENFORCE(argMap["broadcast"].has_i(), "Invalid broadcast argument");
          int broadcast = static_cast<int>(argMap["broadcast"].i());
          c->setBroadcast(!!broadcast);
        }
        return nnOp;
        */
    }
}

register_converter!{Add, AddConverter}

///--------------------------------------------
pub struct CastConverter { }

impl Converter for CastConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::Cast>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::Cast>(nnOp.get());
        ArgumentHelper helper(op);
        c->setTo(cast::GetCastDataType(helper, "to"));
        return nnOp;
        */
    }
}

register_converter!{Cast, CastConverter}

///--------------------------------------------
pub struct ReplaceNaNConverter { }

impl Converter for ReplaceNaNConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::ReplaceNaN>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::ReplaceNaN>(nnOp.get());
        if (argMap.count("value")) {
          CAFFE_ENFORCE(argMap["value"].has_f(), "Invalid 'value' argument");
          float value = static_cast<float>(argMap["value"].f());
          c->setValue(value);
        }
        return nnOp;
        */
    }
}

register_converter!{ReplaceNaN, ReplaceNaNConverter}

///--------------------------------------------
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

///--------------------------------------------
pub struct SliceConverter { }

impl Converter for SliceConverter {
    
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::Slice>();
        const caffe2::ArgumentHelper args(op);

        auto c = dyn_cast<repr::Slice>(nnOp.get());
        if (args.HasArgument("starts")) {
          c->setStarts(args.GetRepeatedArgument<int64_t>("starts"));
        }
        if (args.HasArgument("ends")) {
          c->setEnds(args.GetRepeatedArgument<int64_t>("ends"));
        }
        return nnOp;
        */
    }
    
    #[inline] fn convert_to_operator_def(&mut self, nn_op: *const NeuralNetOperator) -> OperatorDef {
        
        todo!();
        /*
            auto slice = dyn_cast<repr::Slice>(nnOp);
        OperatorDef op;
        op.set_type("Slice");
        op.add_arg()->CopyFrom(
            caffe2::MakeArgument<vector<int64_t>>("starts", slice->getStarts()));
        op.add_arg()->CopyFrom(
            caffe2::MakeArgument<vector<int64_t>>("ends", slice->getEnds()));
        op.mutable_device_option()->CopyFrom(getDeviceOption(nnOp));
        return op;
        */
    }
}

register_converter!{Slice, SliceConverter}

///--------------------------------------------
pub struct ClipRangesGatherSigridHashConverter { }

impl Converter for ClipRangesGatherSigridHashConverter {
    
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::ClipRangesGatherSigridHash>();
        const caffe2::ArgumentHelper args(op);

        auto c = dyn_cast<repr::ClipRangesGatherSigridHash>(nnOp.get());
        if (args.HasArgument("feature_indices")) {
          c->setFeatureIndices(
              args.GetRepeatedArgument<int64_t>("feature_indices"));
        }
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
            auto fuse = dyn_cast<repr::ClipRangesGatherSigridHash>(nnOp);
        OperatorDef op;
        op.set_type("ClipRangesGatherSigridHash");
        op.add_arg()->CopyFrom(caffe2::MakeArgument<vector<int64_t>>(
            "feature_indices", fuse->getFeatureIndices()));
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
    ClipRangesGatherSigridHash,
    ClipRangesGatherSigridHashConverter
}

///--------------------------------------------
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

///--------------------------------------------
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

///--------------------------------------------
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
