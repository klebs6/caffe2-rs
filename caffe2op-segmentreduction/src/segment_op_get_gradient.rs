crate::ix!();

/**
  | base implementation of sorted/unsorted
  | sparse/non-sparse gradient computation
  |
  */
pub struct SegmentOpGetGradient<ForwardOp,ReducerDef,ReducerGradient,const Sorted: bool,const SparseFused: bool> {
    phantomForwardOp:       PhantomData<ForwardOp>,
    phantomReducerDef:      PhantomData<ReducerDef>,
    phantomReducerGradient: PhantomData<ReducerGradient>,
}

impl<
    ForwardOp,
    ReducerDef,
    ReducerGradient,
    const Sorted: bool,
    const SparseFused: bool> 
GetGradientDefs for 
    SegmentOpGetGradient<
        ForwardOp,
        ReducerDef,
        ReducerGradient,
        Sorted,
        SparseFused> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            !ReducerGradient::requiresDataInput(Def()),
            "grads on aux inputs are not yet implemented for Segment operators.");
        vector<string> grad_ins;
        for (const int i : ReducerGradient::originalInputs()) {
          grad_ins.push_back(I(i));
        }
        grad_ins.push_back(GO(0));
        grad_ins.push_back(I(ForwardOp::SEGMENT_IDS));
        vector<OperatorDef> r{CreateOperatorDef(
            string(Sorted ? "SortedSegment" : "UnsortedSegment") +
                ReducerDef::name + "Gradient",
            "",
            grad_ins,
            // no gradient on segment_ids or auxiliary inputs for now
            vector<string>{SparseFused ? GI_V(0) : GI(0)})};
        if (SparseFused) {
          SetSparse(0, I(ForwardOp::INDICES), GI_V(0));
        }
        return r;
        */
    }
}
