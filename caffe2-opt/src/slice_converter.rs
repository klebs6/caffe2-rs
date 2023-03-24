crate::ix!();

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

