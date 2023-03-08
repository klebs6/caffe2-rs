crate::ix!();

/**
  | Mean computes the element-wise mean
  | of the input slices.
  | 
  | Operation doesn't change the shape
  | of the individual blocks.
  |
  */
pub struct MeanReducerDef {
    /*
      template <typename T, class Context>
      using Reducer = MeanReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = MeanReducerGradient<T, Context>;

      static constexpr const char* name = "Mean";
    */

}

impl MeanReducerDef {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
        
        */
    }
}
