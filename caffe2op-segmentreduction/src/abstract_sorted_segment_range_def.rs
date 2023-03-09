crate::ix!();

/**
  | Applies '{op}' to each segment of input
  | tensor. In order to allow for more efficient
  | implementation of '{op}', the input
  | segments have to be contiguous and non-empty.
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
pub struct AbstractSortedSegmentRangeDef<T,SIndex,Context,ReducerDef> {

    /*
    |    pub type OpDef = ReducerDef;
    |
    | using ForwardOp = AbstractSortedSegmentRangeOp<
    |   T,
    |   SIndex,
    |   Context,
    |   typename ReducerDef::template Reducer<T, Context>>;
    |
    | using BackwardOp = AbstractSortedSegmentRangeGradientOp<
    |   T,
    |   SIndex,
    |   Context,
    |   typename ReducerDef::template ReducerGradient<T, Context>>;
    */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomSIndex:     PhantomData<SIndex>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef> AbstractSortedSegmentRangeDef<T,SIndex,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor to be aggregated");
            schema.Input(
                1,
                "SEGMENT_IDS",
                "Vector with the same length as the first dimension of DATA "
                "and values in the range 0..K-1 and in increasing order that "
                "maps each slice of DATA to one of the segments");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated tensor with the first dimension of K and the "
                "other dimentsions inherited from DATA");
        */
    }
}
