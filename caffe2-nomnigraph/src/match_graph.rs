crate::ix!();

/**
  | MatchGraph is a graph of MatchPredicate
  | and it contains utilities for subgraph
  | matching.
  | 
  | (TODO) the subgraph matching methods
  | currently still requires a root match
  | node to be passed in.
  | 
  | We should improve the matching algorithm
  | to eliminate this requirement.
  |
  */
pub struct MatchGraph<G: GraphType> {
    base: NomGraph<MatchPredicate<G>>,
}

impl<G: GraphType> GraphType for MatchGraph<G> {
    type NodeRef = <G as GraphType>::NodeRef;
    type EdgeRef = <G as GraphType>::EdgeRef;

    //TODO: is this the right SubgraphType?
    type SubgraphType = SubgraphType<Self::NodeRef, Self::EdgeRef>;
}

impl<G: GraphType> MatchGraph<G> {

    #[inline] pub fn is_node_match(
        &self, 
        node: <G as GraphType>::NodeRef, 
        match_predicate: &MatchPredicate<G>) -> bool 
    {
        todo!();
        /*
            return matchPredicate.getCriteria()(node);
        */
    }

    /**
      | Check if there can be a subgraph that
      | matches the given criteria that is rooted
      | at the given rootNode.
      | 
      | The flag invertGraphTraversal specify
      | if we should follow out edges or in edges.
      | 
      | The default is true which is useful for
      | a functional interpretation of a dataflow
      | graph.
      |
      */
    #[inline] pub fn is_subgraph_match(&self, 
        root:                   <G as GraphType>::NodeRef,
        root_criteria_ref:      &<MatchGraph<G> as GraphType>::NodeRef,
        invert_graph_traversal: Option<bool>,
        debug:                  Option<bool>) -> SubgraphMatchResultType<G> 
    {
        let invert_graph_traversal: bool = invert_graph_traversal.unwrap_or(true);
        let debug: bool = debug.unwrap_or(false);

        todo!();
        /*
        // Create a matched result that owns a matched subgraph object and pass
        // the subgraph object around to construct it during matching.
        auto matchedResult = SubgraphMatchResultType::matched(true);
        auto result = isSubgraphMatchInternal(
            matchedResult.getMatchNodeMap(),
            matchedResult.getMatchedSubgraph(),
            root,
            rootCriteriaRef,
            rootCriteriaRef->data().shouldIncludeInSubgraph(),
            invertGraphTraversal,
            debug);
        return result.isMatch() ? matchedResult : result;
        */
    }

    /**
      | Utility to transform a graph by looking
      | for subgraphs that match a given pattern
      | and then allow callers to mutate the
      | graph based on subgraphs that are found.
      | 
      | The current implementation doesn't
      | handle any graph transformation itself.
      | 
      | Callers should be responsible for all
      | intended mutation, including deleting
      | nodes in the subgraphs found by this
      | algorithm.
      | 
      | -----------
      | @note
      | 
      | if the replaceFunction lambda returns
      | false, the entire procedure is aborted.
      | 
      | This maybe useful in certain cases when
      | we want to terminate the subgraph search
      | early. invertGraphTraversal flag:
      | see documentation in isSubgraphMatch
      |
      */
    #[inline] pub fn replace_subgraph(&self, 
        graph:                  &mut G,
        criteria:               &<MatchGraph<G> as GraphType>::NodeRef,
        replace_function:       &ReplaceGraphOperation<G>,
        invert_graph_traversal: Option<bool>)  
    {
        let invert_graph_traversal: bool = invert_graph_traversal.unwrap_or(true);

        todo!();
        /*
            for (auto nodeRef : graph.getMutableNodes()) {
          // Make sure the node is still in the graph.
          if (!graph.hasNode(nodeRef)) {
            continue;
          }
          auto matchResult =
              isSubgraphMatch(nodeRef, criteria, invertGraphTraversal);
          if (matchResult.isMatch()) {
            if (!replaceFunction(graph, nodeRef, matchResult)) {
              // If replaceFunction returns false, it means that we should abort
              // the entire procedure.
              break;
            }
          }
        }
        */
    }
    
    #[inline] pub fn is_subgraph_match_internal(&self, 
        matched_nodes:          Arc<MatchNodeMap<G>>,
        matched_subgraph:       Arc<<G as GraphType>::SubgraphType>,
        root:                   <G as GraphType>::NodeRef,
        root_criteria_ref:      &<MatchGraph<G> as GraphType>::NodeRef,
        include_in_subgraph:    bool,
        invert_graph_traversal: bool,
        debug:                  bool) -> SubgraphMatchResultType<G> {

        todo!();
        /*
            auto rootCriteriaNode = rootCriteriaRef->data();

        if (rootCriteriaNode.getCount() == 1) {
          auto matchedNodeEntry = matchedNodes->find(rootCriteriaRef);
          if (matchedNodeEntry != matchedNodes->end()) {
            // If rootCriteriaRef has been matched before (without multiplicity),
            // we should look up the corresponding matched node in the graph
            // and verify if it is the same.
            auto matchedNode = matchedNodeEntry->second;
            if (matchedNode == root) {
              return SubgraphMatchResultType::matched();
            } else if (debug) {
              std::ostringstream debugMessage;
              debugMessage << "Subgraph root at " << root << " is not the same as "
                           << matchedNode << " which previously matched criteria "
                           << debugString(rootCriteriaRef, invertGraphTraversal);
              return SubgraphMatchResultType::notMatched(debugMessage.str());
            } else {
              return SubgraphMatchResultType::notMatched();
            }
          }
        }

        if (!isNodeMatch(root, rootCriteriaNode)) {
          if (debug) {
            std::ostringstream debugMessage;
            debugMessage << "Subgraph root at " << root
                         << " does not match criteria "
                         << debugString(rootCriteriaRef, invertGraphTraversal);
            return SubgraphMatchResultType::notMatched(debugMessage.str());
          } else {
            return SubgraphMatchResultType::notMatched();
          }
        }
        if (rootCriteriaNode.isNonTerminal()) {
          // This is sufficient to be a match if this criteria specifies a non
          // terminal node.
          matchedNodes->emplace(rootCriteriaRef, root);
          if (includeInSubgraph) {
            matchedSubgraph->addNode(root);
          }
          return SubgraphMatchResultType::matched();
        }
        auto& edges =
            invertGraphTraversal ? root->getInEdges() : root->getOutEdges();

        int numEdges = edges.size();
        const auto criteriaEdges = invertGraphTraversal
            ? rootCriteriaRef->getInEdges()
            : rootCriteriaRef->getOutEdges();
        int numChildrenCriteria = criteriaEdges.size();

        // The current algorithm implies that the ordering of the children is
        // important. The children nodes will be matched with the children subgraph
        // criteria in the given order.

        int currentEdgeIdx = 0;
        for (int criteriaIdx = 0; criteriaIdx < numChildrenCriteria;
             criteriaIdx++) {
          auto childrenCriteriaRef = invertGraphTraversal
              ? criteriaEdges[criteriaIdx]->tail()
              : criteriaEdges[criteriaIdx]->head();

          int expectedCount = childrenCriteriaRef->data().getCount();
          bool isStarCount = expectedCount == MatchPredicate<GraphType>::kStarCount;

          int countMatch = 0;

          // Continue to match subsequent edges with the current children criteria.
          // Note that if the child criteria is a * pattern, this greedy algorithm
          // will attempt to find the longest possible sequence that matches the
          // children criteria.
          for (; currentEdgeIdx < numEdges &&
               (isStarCount || countMatch < expectedCount);
               currentEdgeIdx++) {
            auto edge = edges[currentEdgeIdx];
            auto child = invertGraphTraversal ? edge->tail() : edge->head();
            bool shouldIncludeEdgeInSubgraph =
                childrenCriteriaRef->data().shouldIncludeInSubgraph() &&
                includeInSubgraph;

            if (!isSubgraphMatchInternal(
                     matchedNodes,
                     matchedSubgraph,
                     child,
                     childrenCriteriaRef,
                     shouldIncludeEdgeInSubgraph,
                     invertGraphTraversal,
                     debug)
                     .isMatch()) {
              if (!isStarCount) {
                // If the current criteria isn't a * pattern, this indicates a
                // failure.
                if (debug) {
                  std::ostringstream debugMessage;
                  debugMessage << "Child node at " << child
                               << " does not match child criteria "
                               << debugString(
                                      childrenCriteriaRef, invertGraphTraversal)
                               << ". We expected " << expectedCount
                               << " matches but only found " << countMatch << ".";
                  return SubgraphMatchResultType::notMatched(debugMessage.str());
                } else {
                  return SubgraphMatchResultType::notMatched();
                }
              } else {
                // Otherwise, we should move on to the next children criteria.
                break;
              }
            } else if (shouldIncludeEdgeInSubgraph) {
              matchedSubgraph->addEdge(edge);
            }

            countMatch++;
          }

          if (countMatch < expectedCount) {
            // Fails because there are not enough matches as specified by the
            // criteria.
            if (debug) {
              std::ostringstream debugMessage;
              debugMessage << "Expected " << expectedCount
                           << " matches for child criteria "
                           << debugString(childrenCriteriaRef, invertGraphTraversal)
                           << " but only found " << countMatch;
              return SubgraphMatchResultType::notMatched(debugMessage.str());
            } else {
              return SubgraphMatchResultType::notMatched();
            }
          }
        }

        if (currentEdgeIdx < numEdges) {
          // Fails because there are unmatched edges.
          if (debug) {
            std::ostringstream debugMessage;
            debugMessage << "Unmatched children for subgraph root at " << root
                         << ". There are " << numEdges
                         << " children, but only found " << currentEdgeIdx
                         << " matches for the children criteria.";
            return SubgraphMatchResultType::notMatched(debugMessage.str());
          } else {
            return SubgraphMatchResultType::notMatched();
          }
        }
        matchedNodes->emplace(rootCriteriaRef, root);
        if (includeInSubgraph) {
          matchedSubgraph->addNode(root);
        }
        return SubgraphMatchResultType::matched();
        */
    }

    /**
      | TODO: Reuse convertToDotString once
      | convertToDotString can work with subgraph.
      |
      */
    #[inline] pub fn debug_string(
        &self, 
        root_criteria_ref:      <MatchGraph<G> as GraphType>::NodeRef, 
        invert_graph_traversal: bool) -> String 
    {
        todo!();
        /*
            std::ostringstream out;
        auto rootNode = rootCriteriaRef->data();
        out << "{root = '" << rootNode.getDebugString() << "'";
        if (rootNode.getCount() != 1) {
          out << ", count = " << rootNode.getCount();
        }
        if (rootNode.isNonTerminal()) {
          out << ", nonTerminal = " << rootNode.isNonTerminal();
        }
        auto edges = invertGraphTraversal ? rootCriteriaRef->getInEdges()
                                          : rootCriteriaRef->getOutEdges();
        if (!edges.empty()) {
          out << ", childrenCriteria = [";
          for (auto& child : edges) {
            auto nextNode = invertGraphTraversal ? child->tail() : child->head();
            out << debugString(nextNode, invertGraphTraversal) << ", ";
          }
          out << "]";
        }
        out << "}";
        return out.str();
        */
    }
}

