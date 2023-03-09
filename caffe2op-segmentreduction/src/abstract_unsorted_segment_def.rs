crate::ix!();

/**
  | Applies '{op}' to each segment of input
  | tensor. Segments ids can appear in arbitrary
  | order (unlike in SortedSegment{op}).
  | 
  | SEGMENT_IDS is a vector that maps each
  | of the first dimension slices of the
  | 
  | DATA to a particular group (segment).
  | Values belonging to the same segment
  | are aggregated together.
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
pub struct AbstractUnsortedSegmentDef<T,SIndex,Context,ReducerDef> {
    
    /*
      | using OpDef = ReducerDef;
      |
      | static constexpr const char* basename = "UnsortedSegment";
      |
      | using Reducer = typename ReducerDef::template Reducer<T, Context>;
      |
      | using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractUnsortedSegmentOp<
      |     T,
      |     SIndex,
      |     Context,
      |     typename ReducerDef::template Reducer<T, Context>,
      |     false>;
      |
      | using BackwardOp =
      |     AbstractUnsortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
      |
      | using GetGradient = SegmentOpGetGradient<
      |     ForwardOp,
      |     ReducerDef,
      |     ReducerGradient,
      |     false /*Sorted*/,
      |     false /*SparseFused*/>;
    */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomSIndex:     PhantomData<SIndex>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef> AbstractUnsortedSegmentDef<T,SIndex,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Arg(
                "num_segments",
                "Optional int argument specifying the number of output segments and "
                "thus the first dimension of the output");
            schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
            schema.Input(
                <R as Reducer>::InputCount,
                "SEGMENT_IDS",
                "Integer vector with the same length as the first dimension of DATA "
                "that maps each slice of DATA to one of the segments");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of equal to the "
                "number of segments.");
            ReducerDef::PopulateSchema(schema);
        */
    }
}
