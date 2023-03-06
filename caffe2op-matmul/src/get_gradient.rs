crate::ix!();

pub struct GetMatMulGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMatMulGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(def_.input_size() == 2 || def_.input_size() == 3);

        bool axis_a = 1;
        bool axis_b = 1;
        bool trans_a = 0;
        bool trans_b = 0;

        if (ArgumentHelper::HasArgument(Def(), "trans_a")) {
          trans_a = GetArgument(Def(), "trans_a").i();
        }
        if (ArgumentHelper::HasArgument(Def(), "trans_b")) {
          trans_b = GetArgument(Def(), "trans_b").i();
        }
        if (ArgumentHelper::HasArgument(Def(), "axis_a")) {
          axis_a = GetArgument(Def(), "axis_a").i();
        }
        if (ArgumentHelper::HasArgument(Def(), "axis_b")) {
          axis_b = GetArgument(Def(), "axis_b").i();
        }

        if (trans_a) {
          if (trans_b) {
            // A'B':
            // dA = B'G', dB = G'A'
            return vector<OperatorDef>{
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{I(1), GO(0), I(0)},
                    vector<string>{GI(0)},
                    vector<Argument>{MakeArgument<int>("trans_a", 1),
                                     MakeArgument<int>("trans_b", 1),
                                     MakeArgument<int>("axis_a", axis_b)}),
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{GO(0), I(0), I(1)},
                    vector<string>{GI(1)},
                    vector<Argument>{MakeArgument<int>("trans_a", 1),
                                     MakeArgument<int>("trans_b", 1),
                                     MakeArgument<int>("axis_b", axis_a)})};
          } else {
            // A'B:
            // dA = BG', dB = AG
            return vector<OperatorDef>{
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{I(1), GO(0), I(0)},
                    vector<string>{GI(0)},
                    vector<Argument>{MakeArgument<int>("trans_b", 1),
                                     MakeArgument<int>("axis_a", axis_b)}),
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{I(0), GO(0), I(1)},
                    vector<string>{GI(1)},
                    vector<Argument>{MakeArgument<int>("axis_a", axis_a)})};
          }
        } else {
          if (trans_b) {
            // AB':
            // dA = GB, dB = G'A
            return vector<OperatorDef>{
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{GO(0), I(1), I(0)},
                    vector<string>{GI(0)},
                    vector<Argument>{MakeArgument<int>("axis_b", axis_b)}),
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{GO(0), I(0), I(1)},
                    vector<string>{GI(1)},
                    vector<Argument>{MakeArgument<int>("trans_a", 1),
                                     MakeArgument<int>("axis_b", axis_a)})};
          } else {
            // AB:
            // dA = GB', dB = A'G
            return vector<OperatorDef>{
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{GO(0), I(1), I(0)},
                    vector<string>{GI(0)},
                    vector<Argument>{MakeArgument<int>("trans_b", 1),
                                     MakeArgument<int>("axis_b", axis_b)}),
                CreateOperatorDef(
                    "MatMul",
                    "",
                    vector<string>{I(0), GO(0), I(1)},
                    vector<string>{GI(1)},
                    vector<Argument>{MakeArgument<int>("trans_a", 1),
                                     MakeArgument<int>("axis_a", axis_a)})};
          }
        }
        */
    }
}

impl<'a> CopyArguments for GetMatMulGradient<'a> {
    
    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_gradient!{MatMul, GetMatMulGradient}
