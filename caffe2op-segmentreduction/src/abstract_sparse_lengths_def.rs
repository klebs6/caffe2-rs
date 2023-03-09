crate::ix!();

/**
  | Pulls in slices of the input tensor,
  | groups them into segments and applies
  | '{op}' to each segment. Segments are
  | defined by their LENGTHS.
  | 
  | This op is basically Gather and Lengths{op}
  | fused together.
  | 
  | INDICES should contain integers in
  | range 0..N-1 where N is the first dimension
  | of DATA. INDICES represent which slices
  | of DATA need to be pulled in.
  | 
  | LENGTHS is a vector that defines slice
  | sizes by first dimension of DATA. Values
  | belonging to the same segment are aggregated
  | together. sum(LENGTHS) has to match
  | INDICES size.
  | 
  | The first dimension of the output is
  | equal to the number of input segment,
  | i.e. `len(LENGTHS)`. Other dimensions
  | are inherited from the input tensor.
  | 
  | {op_doc} bool GradientNeedIndices
  | = false>
  |
  */
pub struct AbstractSparseLengthsDef<T,SIndex,Context,ReducerDef,const GradientNeedIndices: bool> {
    
    /*
      | using OpDef = ReducerDef;
      |
      | static constexpr const char* basename = "SparseLengths";
      |
      | using Reducer = typename ReducerDef::template Reducer<T, Context>;
      |
      | using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractLengthsOp<T, SIndex, Context, Reducer>;
      |
      | // TODO(dzhulgakov): we're registering the same class twice here,
      | // consider avoiding op duplication here
      | // Note: registering 2 input version for now because of naming in the macro,
      | // will register 3 input version alone
      | /* INDICES are not used in CPU version, but they are needed in async CUDA
      |  *    version. So we register 3 input version for CPU as gradient op for
      |  *    GPU/CPU convert. We then register 2 input version for CPU for backward
      |  *    compatibility with older nets.
      |  */
      | using BackwardOp = AbstractLengthsGradientOp<
      |     T,
      |     SIndex,
      |     Context,
      |     ReducerGradient,
      |     false /*GradientNeedIndices*/>;
      |
      | using WithMainInputBackwardOp = AbstractLengthsWithMainInputGradientOp<
      |     T,
      |     T,
      |     SIndex,
      |     Context,
      |     ReducerGradient>;
      |
      | // Will return 3 input version. This is aligning new CPU/GPU nets.
      | using GetGradient = LengthsOpGetGradient<
      |     ForwardOp,
      |     ReducerDef,
      |     ReducerGradient,
      |     true /*SparseFused*/,
      |     GradientNeedIndices>;
    */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomSIndex:     PhantomData<SIndex>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef,const GradientNeedIndices: bool> 
AbstractSparseLengthsDef<T,SIndex,Context,ReducerDef,GradientNeedIndices> {

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
                "LENGTHS",
                "Non negative vector with sum of elements equal to INDICES length");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of K "
                "(the number of segments).");
            schema.TensorInferenceFunction(OpSchema::NeedsAllInputShapes(
                [](const OperatorDef&, const std::vector<TensorShape>& input_types) {
                  std::vector<TensorShape> out(1);
                  out[0] = input_types[0];
                  out[0].set_dims(0, input_types[<R as Reducer>::InputCount + 1].dims(0));
                  return out;
                }));
            ReducerDef::PopulateSchema(schema);

            schema.CostInferenceFunction(
                [](const OperatorDef& def,
                   const vector<TensorShape>& inputs) -> OpSchema::Cost {
                  return CostInferenceForSparseLengths(
                      def, inputs, strcmp(OpDef::name, "WeightedSum") == 0);
                });
        */
    }
}
