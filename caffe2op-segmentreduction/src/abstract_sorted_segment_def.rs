crate::ix!();

/**
  | Applies '{op}' to each segment of input
  | tensor. Segments need to be sorted and
  | contiguous. See also UnsortedSegment{op}
  | that doesn't have this requirement.
  | 
  | SEGMENT_IDS is a vector that maps each
  | of the first dimension slices of the
  | 
  | DATA to a particular group (segment).
  | Values belonging to the same segment
  | are aggregated together.
  | 
  | The first dimension of the output is
  | equal to the number of input segments,
  | i.e. `SEGMENT_IDS[-1]+1`. Other dimensions
  | are inherited from the input tensor.
  | 
  | {op_doc}
  |
  */
pub struct AbstractSortedSegmentDef<T,SIndex,Context,ReducerDef> {
    
    /*
      | using OpDef = ReducerDef;
      |
      | static constexpr const char* basename = "SortedSegment";
      |
      | using Reducer = typename ReducerDef::template Reducer<T, Context>;
      |
      | using ReducerGradient =
      |     typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractSortedSegmentOp<T, SIndex, Context, Reducer, false>;
      |
      | using BackwardOp =
      |     AbstractSortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
      |
      | using GetGradient = SegmentOpGetGradient<
      |     ForwardOp,
      |     ReducerDef,
      |     ReducerGradient,
      |     true /*Sorted*/,
      |     false /*SparseFused*/>;
    */
        phantom:           PhantomData<Context>,
        phantomT:          PhantomData<T>,
        phantomSIndex:     PhantomData<SIndex>,
        phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef> AbstractSortedSegmentDef<T,SIndex,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
            schema.Input(
                <R as Reducer>::InputCount,
                "SEGMENT_IDS",
                "Vector with the same length as the first dimension of DATA "
                "and values in the range 0..K-1 and in increasing order that "
                "maps each slice of DATA to one of the segments");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of K "
                "(the number of segments).");
            ReducerDef::PopulateSchema(schema);
        */
    }
}
