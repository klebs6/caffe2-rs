crate::ix!();

/**
  | Pulls in slices of the input tensor,
  | groups them into segments and applies
  | '{op}' to each segment. Segments need
  | to be sorted and contiguous. See also
  | 
  | SparseUnsortedSegment{op} that doesn't
  | have this requirement.
  | 
  | This op is basically Gather and SortedSegment{op}
  | fused together.
  | 
  | INDICES should contain integers in
  | range 0..N-1 where N is the first dimension
  | of DATA. INDICES represent which slices
  | of DATA need to be pulled in.
  | 
  | SEGMENT_IDS is a vector that maps each
  | referenced slice of the DATA to a particular
  | group (segment). Values belonging
  | to the same segment are aggregated together.
  | SEGMENT_IDS should have the same dimension
  | as INDICES.
  | 
  | The first dimension of the output is
  | equal to the number of input segments,
  | i.e. `SEGMENT_IDS[-1]+1`. Other dimensions
  | are inherited from the input tensor.
  | 
  | {op_doc}
  |
  */
pub struct AbstractSparseSortedSegmentDef<T,SIndex,Context,ReducerDef> {
    
    /*
      | using OpDef = ReducerDef;
      |
      | static constexpr const char* basename = "SparseSortedSegment";
      |
      | using Reducer = typename ReducerDef::template Reducer<T, Context>;
      |
      | using ReducerGradient =
      |     typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractSortedSegmentOp<T, SIndex, Context, Reducer>;
      |
      | // TODO(dzhulgakov): we're registering the same class twice here,
      | // consider avoiding op duplication here
      | using BackwardOp = AbstractSortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
      |
      | using GetGradient = SegmentOpGetGradient<
      |     ForwardOp,
      |     ReducerDef,
      |     ReducerGradient,
      |     true /*Sorted*/,
      |     true /*SparseFused*/>;
    */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomSIndex:     PhantomData<SIndex>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef> AbstractSparseSortedSegmentDef<T,SIndex,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
            schema.Input(
                <R as Reducer>::InputCount,
                "INDICES",
                "Integer vector containing indices of the first dimension of DATA for "
                "the slices that are being aggregated");
            schema.Input(
                <R as Reducer>::InputCount + 1,
                "SEGMENT_IDS",
                "Vector with the same length as INDICES and values in the range "
                "0..K-1 and in increasing order that maps each slice of DATA referenced"
                " by INDICES to one of the segments");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of K "
                "(the number of segments).");
            ReducerDef::PopulateSchema(schema);
        */
    }
}
