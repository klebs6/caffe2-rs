crate::ix!();

#[inline] pub fn op_registry_key(
    op_type: &String,
    engine: &String) -> String 
{
    todo!();
    /*
        if (engine == "" || engine == "DEFAULT") {
        return op_type;
      } else {
        return op_type + "_ENGINE_" + engine;
      }
    */
}
