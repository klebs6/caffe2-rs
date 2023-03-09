crate::ix!();

/**
  | Pulls in slices of the input tensor,
  | groups them into segments and applies
  | '{op}' to each segment. Segments ids
  | can appear in arbitrary order (unlike
  | in
  | 
  | SparseSortedSegment{op}).
  | 
  | This op is basically Gather and UnsortedSegment{op}
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
  | If `num_segments` argument is passed
  | it would be used as a first dimension
  | for the output. Otherwise, it'd be dynamically
  | calculated from as the max value of
  | 
  | SEGMENT_IDS plus one. Other output
  | dimensions are inherited from the input
  | tensor.
  | 
  | {op_doc}
  |
  */
pub struct AbstractSparseUnsortedSegmentDef<T,SIndex,Context,ReducerDef> {
    
    /*
    | using OpDef = ReducerDef;
    |
    | static constexpr const char* basename = "SparseUnsortedSegment";
    |
    | using Reducer = typename ReducerDef::template Reducer<T, Context>;
    |
    | using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;
    |
    | using ForwardOp = AbstractUnsortedSegmentOp<T, SIndex, Context, Reducer>;
    | // TODO(dzhulgakov): we're registering the same class twice here,
    | // consider avoiding op duplication here
    | using BackwardOp = AbstractUnsortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
    |
    | using GetGradient = SegmentOpGetGradient<
    |     ForwardOp,
    |     ReducerDef,
    |     ReducerGradient,
    |     false /*Sorted*/,
    |     true /*SparseFused*/>;
        */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomSIndex:     PhantomData<SIndex>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef> AbstractSparseUnsortedSegmentDef<T,SIndex,Context,ReducerDef> {

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
                "Integer vector with the same length as INDICES that maps each slice "
                "of DATA referenced by INDICES to one of the segments");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of equal to the "
                "number of segments.");
            ReducerDef::PopulateSchema(schema);
        */
    }
}
