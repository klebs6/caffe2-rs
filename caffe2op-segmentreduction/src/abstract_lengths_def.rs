crate::ix!();

/**
  | Applies '{op}' to each segment of the
  | input tensor. Segments are defined
  | by their *LENGTHS*. *LENGTHS* is a vector
  | that maps each of the slices of
  | 
  | DATA* to a particular segment. Values
  | belonging to the same segment are aggregated
  | together and considered for the '{op}'
  | operation.
  | 
  | For example *LENGTHS = [2, 1]* stands
  | for segments *DATA[0..1]* and *DATA[2]*
  | 
  | The sum of elements in *LENGTHS* must
  | equal the number of elements in the first
  | dimension of *DATA*. The length of *OUTPUT*
  | is equal to the number of input segments,
  | i.e. len(*LENGTHS*).
  | 
  | {op_doc}
  | 
  | {extra}
  | 
  | bool GradientNeedIndices = false>
  |
  */
pub struct AbstractLengthsDef<T,SIndex,Context,ReducerDef,const GradientNeedIndices: bool> {
    
    /*
      using OpDef = ReducerDef;

      static constexpr const char* basename = "Lengths";

      using Reducer = typename ReducerDef::template Reducer<T, Context>;

      using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;

      using ForwardOp = AbstractLengthsOp<T, SIndex, Context, Reducer, false>;

      using BackwardOp = AbstractLengthsGradientOp<T, SIndex, Context, ReducerGradient>;

      using WithMainInputBackwardOp = AbstractLengthsWithMainInputGradientOp<
          T,
          T,
          SIndex,
          Context,
          ReducerGradient,
          false>;

      using WithMainInputAndForwardOutputBackwardOp =
          AbstractLengthsWithMainInputAndForwardOutputGradientOp<
              T,
              SIndex,
              Context,
              ReducerGradient>;

      using GetGradient = LengthsOpGetGradient<
          ForwardOp,
          ReducerDef,
          ReducerGradient,
          false /*SparseFused*/,
          GradientNeedIndices>;
    */
        phantom: PhantomData<Context>,
        phantomT: PhantomData<T>,
        phantomSIndex: PhantomData<SIndex>,
        phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef,const GradientNeedIndices: bool> 
AbstractLengthsDef<T,SIndex,Context,ReducerDef,GradientNeedIndices> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
            schema.Input(
                <R as Reducer>::InputCount,
                "LENGTHS",
                "Vector with the same sum of elements as the first dimension of DATA");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of len(LENGTHS) ");
            schema.TensorInferenceFunction(
                [](const OperatorDef& def, const vector<TensorShape>& in) {
                  vector<TensorShape> out(0);
                  TensorShape output;
                  for (int d : in[<R as Reducer>::InputCount].dims()) {
                    output.add_dims(d);
                  }
                  for (int j = 1; j < in[0].dims_size(); j++) {
                    output.add_dims(in[0].dims(j));
                  }
                  output.set_data_type(in[0].data_type());
                  out.push_back(output);
                  return out;
                });
            ReducerDef::PopulateSchema(schema);
        */
    }
}


