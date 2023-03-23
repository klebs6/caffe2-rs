crate::ix!();

pub type SubgraphMatchResultType<G: GraphType> = SubgraphMatchResult<G>;

pub type ReplaceGraphOperation<G: GraphType> = fn(
    _u0: &mut G,
    _u1: <G as GraphType>::NodeRef,
    _u2: &SubgraphMatchResultType<G>
) -> bool;

/**
  | Map from match node to corresponding
  | node in the graph to be scanned.
  |
  */
pub type MatchNodeMap<G: GraphType> 
= HashMap<<MatchGraph<G> as GraphType>::NodeRef, <G as GraphType>::NodeRef>;

pub struct SubgraphMatchResult<G: GraphType> {
    is_match:            bool,
    debug_message:       String,
    matched_subgraph:    Arc<<G as GraphType>::SubgraphType>,
    match_node_map:      Arc<MatchNodeMap<G>>,
}

impl<G: GraphType> SubgraphMatchResult<G> {

    #[inline] pub fn not_matched_with_debug_msg(debug_message: &String) -> SubgraphMatchResult<G> {
        
        todo!();
        /*
            return SubgraphMatchResult<G>(false, debugMessage);
        */
    }
    
    #[inline] pub fn not_matched() -> SubgraphMatchResult<G> {
        
        todo!();
        /*
            return SubgraphMatchResult<G>(
            false, "Debug message is not enabled");
        */
    }
    
    #[inline] pub fn matched(own_subgraph: Option<bool>) -> SubgraphMatchResult<G> {

        let own_subgraph: bool = own_subgraph.unwrap_or(false);

        todo!();
        /*
            return SubgraphMatchResult<G>(true, "Matched", ownSubgraph);
        */
    }
    
    #[inline] pub fn is_match(&self) -> bool {
        
        todo!();
        /*
            return isMatch_;
        */
    }
    
    #[inline] pub fn get_debug_message(&self) -> String {
        
        todo!();
        /*
            return debugMessage_;
        */
    }
    
    #[inline] pub fn get_matched_subgraph(&self) -> Arc<<G as GraphType>::SubgraphType> {
        
        todo!();
        /*
            return matchedSubgraph_;
        */
    }
    
    #[inline] pub fn get_match_node_map(&self) -> Arc<MatchNodeMap<G>> {
        
        todo!();
        /*
            return matchNodeMap_;
        */
    }
    
    pub fn new(
        is_match:      bool,
        debug_message: &String,
        own_subgraph:  Option<bool>) -> Self {

        let own_subgraph: bool = own_subgraph.unwrap_or(false);

        todo!();
        /*
            : isMatch_(isMatch),
            debugMessage_(debugMessage),
            matchedSubgraph_( ownSubgraph ? std::make_shared<typename G::SubgraphType>() : nullptr),
            matchNodeMap_( ownSubgraph ? std::make_shared<MatchNodeMap>() : nullptr)
        */
    }
}
