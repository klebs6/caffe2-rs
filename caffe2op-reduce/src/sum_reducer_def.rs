crate::ix!();

/**
  | Summation is done element-wise across
  | slices of the input tensor and doesn't
  | change the shape of the individual blocks.
  |
  */
pub struct SumReducerDef {

    /*
      template <typename T, class Context>
      using Reducer = SumReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = SumReducerGradient<T, Context>;

      static constexpr const char* name = "Sum";
    */
}

impl SumReducerDef {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
        
        */
    }
}
