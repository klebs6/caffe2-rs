crate::ix!();

pub struct GetBatchMatMulGradient;

impl GetGradientDefs for GetBatchMatMulGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(def_.input_size(), 2);

        bool broadcast = false;
        if (ArgumentHelper::HasArgument(Def(), "broadcast")) {
          broadcast = GetArgument(Def(), "broadcast").i();
        }
        CAFFE_ENFORCE(
            !broadcast,
            "Gradient is currently not supported with "
            "broadcast=1 for BatchMatMul.");

        bool trans_a = 0;
        bool trans_b = 0;

        if (ArgumentHelper::HasArgument(Def(), "trans_a")) {
          trans_a = GetArgument(Def(), "trans_a").i();
        }
        if (ArgumentHelper::HasArgument(Def(), "trans_b")) {
          trans_b = GetArgument(Def(), "trans_b").i();
        }

        auto no_trans_arg = vector<Argument>();
        auto trans_a_arg = vector<Argument>{MakeArgument<int>("trans_a", 1)};
        auto trans_b_arg = vector<Argument>{MakeArgument<int>("trans_b", 1)};
        auto trans_both_arg = vector<Argument>{MakeArgument<int>("trans_a", 1),
                                               MakeArgument<int>("trans_b", 1)};

        if (trans_a) {
          if (trans_b) {
            // A'B':
            // dA = B'G', dB = G'A'
            return vector<OperatorDef>{CreateOperatorDef(
                                           "BatchMatMul",
                                           "",
                                           vector<string>{I(1), GO(0)},
                                           vector<string>{GI(0)},
                                           trans_both_arg),
                                       CreateOperatorDef(
                                           "BatchMatMul",
                                           "",
                                           vector<string>{GO(0), I(0)},
                                           vector<string>{GI(1)},
                                           trans_both_arg)};
          } else {
            // A'B:
            // dA = BG', dB = AG
            return vector<OperatorDef>{CreateOperatorDef(
                                           "BatchMatMul",
                                           "",
                                           vector<string>{I(1), GO(0)},
                                           vector<string>{GI(0)},
                                           trans_b_arg),
                                       CreateOperatorDef(
                                           "BatchMatMul",
                                           "",
                                           vector<string>{I(0), GO(0)},
                                           vector<string>{GI(1)},
                                           no_trans_arg)};
          }
        } else {
          if (trans_b) {
            // AB':
            // dA = GB, dB = G'A
            return vector<OperatorDef>{CreateOperatorDef(
                                           "BatchMatMul",
                                           "",
                                           vector<string>{GO(0), I(1)},
                                           vector<string>{GI(0)},
                                           no_trans_arg),
                                       CreateOperatorDef(
                                           "BatchMatMul",
                                           "",
                                           vector<string>{GO(0), I(0)},
                                           vector<string>{GI(1)},
                                           trans_a_arg)};
          } else {
            // AB:
            // dA = GB', dB = A'G
            return vector<OperatorDef>{CreateOperatorDef(
                                           "BatchMatMul",
                                           "",
                                           vector<string>{GO(0), I(1)},
                                           vector<string>{GI(0)},
                                           trans_b_arg),
                                       CreateOperatorDef(
                                           "BatchMatMul",
                                           "",
                                           vector<string>{I(0), GO(0)},
                                           vector<string>{GI(1)},
                                           trans_a_arg)};
          }
        }
        */
    }
}

impl CopyArguments for GetBatchMatMulGradient {

    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_gradient!{BatchMatMul, GetBatchMatMulGradient}

