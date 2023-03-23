crate::ix!();

/// \brief Helper for dominator tree finding.
#[inline] pub fn reachable<G: GraphType>(
    root:    <G as GraphType>::NodeRef,
    ignored: <G as GraphType>::NodeRef,
    seen:    *mut HashSet<<G as GraphType>::NodeRef>)  {

    todo!();
    /*
        seen->insert(root);
      for (const auto& outEdge : root->getOutEdges()) {
        auto& newNode = outEdge->head();
        if (newNode != ignored && (seen->find(newNode) == seen->end())) {
          reachable<G>(newNode, ignored, seen);
        }
      }
    */
}
