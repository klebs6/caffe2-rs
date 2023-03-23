crate::ix!();

pub struct NodeHelper {
    g: *mut NNGraph,
}

pub type NNMatchGraph = MatchGraph<NNGraph>;

pub type NNMatchPredicate = MatchPredicate<NNGraph>;

/* ---------- Commonly used node predicate  ---------- */

#[inline] pub fn match_external_tensor_node() -> NNMatchPredicate {
    
    todo!();
    /*
    
    */
}

/**
  | This file defines utilities for matching
  | subgraphs.
  |
  */
pub struct Match<'g, G: GraphType, EqualityClass> {
    match_graph:     &'g mut G,
    match_node_list: Vec<<G as GraphType>::NodeRef>,
    phantom:         PhantomData<EqualityClass>,
}

impl<'g, G: GraphType, EqualityClass> Match<'g, G, EqualityClass> {
    
    pub fn new(g: &'g mut G) -> Self {
        todo!();
        /*
            : MatchGraph(g) 
        // First we sort both the matching graph topologically.
        // This could give us a useful anchor in the best case.
        auto result = nom::algorithm::topoSort(&MatchGraph);
        MatchNodeList = result.nodes;
        */
    }
    
    #[inline] pub fn recursive_match<T,U>(
        &mut self, 
        candidate_node:   <G as GraphType>::NodeRef,
        stack:            Vec<<G as GraphType>::NodeRef>,
        current_subgraph: SubgraphType<T,U>) -> Vec<SubgraphType<T,U>> 
    {
        todo!();
        /*
            if (EqualityClass::equal(stack.back(), candidateNode)) {
          currentSubgraph.addNode(candidateNode);

          // Base case
          if (stack.size() == MatchNodeList.size()) {
            return std::vector<SubgraphType>{currentSubgraph};
          }

          // Recurse and accumulate matches
          stack.emplace_back(MatchNodeList.at(stack.size()));

          std::vector<SubgraphType> matchingSubgraphs;
          for (auto outEdge : candidateNode->getOutEdges()) {
            for (auto subgraph :
                 recursiveMatch(outEdge->head(), stack, currentSubgraph)) {
              matchingSubgraphs.emplace_back(subgraph);
            }
          }
          return matchingSubgraphs;
        }

        // No match here, early bailout
        return std::vector<SubgraphType>{};
        */
    }
    
    #[inline] pub fn match_<T,U>(&mut self, g: &'g mut G) -> Vec<SubgraphType<T,U>> {
        
        todo!();
        /*
            std::vector<SubgraphType> out;

        std::vector<typename G::NodeRef> stack;
        stack.emplace_back(MatchNodeList.front());

        // Try each node in the candidate graph as the anchor.
        for (auto n : g.getMutableNodes()) {
          for (auto subgraph : recursiveMatch(n, stack, SubgraphType())) {
            out.emplace_back(subgraph);
          }
        }

        return out;
        */
    }
}
