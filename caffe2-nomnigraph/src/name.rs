crate::ix!();

/**
  | Get the name of the node regardless of
  | underlying type.
  |
  */
#[inline] pub fn get_name<T,U>(n: NodeRef<T,U>) -> String {
    
    todo!();
    /*
        if (is<NeuralNetData>(n)) {
        return nn::get<NeuralNetData>(n)->getName();
      } else if (is<NeuralNetOperator>(n)) {
        return nn::get<NeuralNetOperator>(n)->getName();
      }
      return "Unknown";
    */
}
