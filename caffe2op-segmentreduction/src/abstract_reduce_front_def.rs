crate::ix!();

/**
  | Reduces the input tensor along the first
  | dimension of the input tensor by applying
  | '{op}'. This op acts in a similar way
  | to SortedSegment{op} and
  | 
  | UnsortedSegment{op} but as if all input
  | slices belong to a single segment.
  | 
  | {op_doc}
  |
  */
pub struct AbstractReduceFrontDef<T,Context,ReducerDef> {
    
    /*
      | using OpDef = ReducerDef;
      |
      | static constexpr const char* basename = "ReduceFront";
      |
      | using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractReduceFrontOrBackOp<
      |     T,
      |     Context,
      |     typename ReducerDef::template Reducer<T, Context>,
      |     true>;
      |
      | using BackwardOp = AbstractReduceFrontOrBackGradientOp<T, Context, ReducerGradient, true>;
      |
    */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,Context,ReducerDef> AbstractReduceFrontDef<T,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(
                0, "DATA", "Input tensor to be reduced on the first dimension");
            schema.TensorInferenceFunction([](const OperatorDef& def,
                                              const vector<TensorShape>& in) {
              CAFFE_ENFORCE_EQ(1, in.size());
              ArgumentHelper helper(def);
              int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1);
              typename ReducerDef::template Reducer<T, Context>::Meta ctx(true);
              vector<int64_t> out_dims = ctx.getOutputShape(in[0], num_reduce_dims);
              return vector<TensorShape>{
                  CreateTensorShape(out_dims, in[0].data_type())};
            });
            ReducerDef::PopulateSchema(schema);
        */
    }
}
