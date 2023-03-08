crate::ix!();

/**
  | Input slices are first scaled by SCALARS
  | and then summed element-wise.
  | 
  | It doesn't change the shape of the individual
  | blocks.
  |
  */
pub struct WeightedSumReducerDef {
    
    /*
      template <typename T, class Context>
      using Reducer = WeightedSumReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = WeightedSumReducerGradient<T, Context>;

      static constexpr const char* name = "WeightedSum";
    */
}

impl WeightedSumReducerDef {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor for the summation");
            schema.Input(
                1,
                "SCALARS",
                "Scalar multipliers for the input slices. Must be a vector with the "
                "length matching the number of slices");
            schema.Arg(
                "grad_on_weights",
                "Produce also gradient for `weights`. For now it's only supported in "
                "`Lengths`-based operators");
        */
    }
}
