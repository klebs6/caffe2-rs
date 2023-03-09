crate::ix!();

/**
  | base implementation of sparse/non-sparse
  | gradient computation
  |
  | bool GradientNeedIndices = false>
  |
  */
pub struct LengthsOpGetGradient<ForwardOp,ReducerDef,ReducerGradient,const SparseFused: bool,const GradientNeedIndices: bool> {
    phantomForwardOp:       PhantomData<ForwardOp>,
    phantomReducerDef:      PhantomData<ReducerDef>,
    phantomReducerGradient: PhantomData<ReducerGradient>,
}

impl<
    ForwardOp,
    ReducerDef,
    ReducerGradient,
    const SparseFused: bool,
    const GradientNeedIndices: bool> 
GetGradientDefs 
    for LengthsOpGetGradient<
        ForwardOp,
        ReducerDef,
        ReducerGradient,
        SparseFused,
        GradientNeedIndices> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_ins;
        string suffix = "Gradient";
        for (const int i : ReducerGradient::originalInputs()) {
          grad_ins.push_back(I(i));
        }
        if (ReducerGradient::requiresForwardOutput()) {
          grad_ins.push_back(O(0));
          CAFFE_ENFORCE(
              !SparseFused,
              "Forward pass output not yet supported as input for backward pass "
              "for SparseLengthsXXX operators");
          suffix = "AndForwardOutput" + suffix;
        }
        grad_ins.push_back(GO(0));
        grad_ins.push_back(I(ForwardOp::LENGTHS));
        bool indices_pushed = false;
        if (ReducerGradient::requiresDataInput(Def())) {
          grad_ins.push_back(I(0));
          if (SparseFused) {
            grad_ins.push_back(I(ForwardOp::INDICES));
            indices_pushed = true;
          }
          suffix = "WithMainInput" + suffix;
        }
        if (GradientNeedIndices && !indices_pushed) {
          if (SparseFused) {
            grad_ins.push_back(I(ForwardOp::INDICES));
          } else {
            // Hacky: using Input as Indices, remove this after we have specialized
            // cuda LengthsIndicesInGradientSumGradient
            grad_ins.push_back(I(0));
          }
        }
        vector<string> grad_outs;
        grad_outs.push_back({SparseFused ? GI_V(0) : GI(0)});
        int aux_grads = ReducerGradient::numAuxInputsWithGrads(Def());
        for (int i = 1; i <= aux_grads; ++i) {
          grad_outs.push_back(GI(i));
        }
        vector<OperatorDef> r{CreateOperatorDef(
            string(SparseFused ? "SparseLengths" : "Lengths") +
                string(GradientNeedIndices ? "IndicesInGradient" : "") +
                ReducerDef::name + suffix,
            "",
            grad_ins,
            grad_outs)};
        if (SparseFused) {
          SetSparse(0, I(ForwardOp::INDICES), GI_V(0));
        }
        return r;
        */
    }
}


