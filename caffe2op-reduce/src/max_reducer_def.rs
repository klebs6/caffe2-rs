crate::ix!();

/**
  | Max computes the element-wise max of
  | the input slices.
  | 
  | Operation doesn't change the shape
  | of the individual blocks.
  |
  */
pub struct MaxReducerDef {
    
    /*
      template <typename T, class Context>
      using Reducer = MaxReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = MaxReducerGradient<T, Context>;

      static constexpr const char* name = "Max";
    */
}

impl MaxReducerDef {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
        
        */
    }
}
