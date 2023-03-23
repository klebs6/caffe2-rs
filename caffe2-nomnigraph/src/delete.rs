crate::ix!();

/// \brief Deletes a referenced node from the control flow graph.
///
#[inline] pub fn delete_node<G: GraphType>(
    cfg: *mut ControlFlowGraph<G>, 
    node: <G as GraphType>::NodeRef)  {

    todo!();
    /*
        for (auto bbNode : cfg->getMutableNodes()) {
        auto bb = bbNode->data().get();
        if (bb->hasInstruction(node)) {
          bb->deleteInstruction(node);
        }
      }
    */
}
