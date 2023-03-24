crate::ix!();

#[inline] pub fn collect_inputs_and_outputs(
    op:      &OperatorDef,
    inputs:  *mut HashSet<String>,
    outputs: *mut HashSet<String>)  {
    
    todo!();
    /*
        for (const auto& blob : op.input()) {
        inputs->emplace(blob);
      }
      for (const auto& blob : op.output()) {
        outputs->emplace(blob);
      }
    */
}

