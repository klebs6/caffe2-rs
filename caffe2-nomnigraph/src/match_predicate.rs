crate::ix!();

pub type Predicate<G: GraphType> = fn(_u0: <G as GraphType>::NodeRef) -> bool;

pub const kStarCount: i32 = -1;

/**
  | MatchGraph is a graph of MatchPredicate.
  | 
  | MatchPredicate needs a predicate for
  | node matching and
  | 
  | - includeInSubgraph: whether this
  | node and nodes/edges reachable from
  | it should be included in the return matched
  | subgraph (if the pattern matches).
  | 
  | This is useful in case we would like to
  | specify a matching pattern but do not
  | want part of the pattern to be included
  | in the returned subgraph.
  | 
  | - A count, which means we may want to match
  | this node multiple times from its incoming
  | edges. The count can be unlimited (think
  | about it as a regex star).
  | 
  | - If nonTerminal flag is set, it means
  | we will not consider outgoing edges
  | from the node when doing subgraph matching.
  |
  */
pub struct MatchPredicate<G: GraphType> {

    criteria:            Predicate<G>,
    count:               i32, // default = 1
    include_in_subgraph: bool, // default = true
    non_terminal:        bool, // default = false
    debug_string:        String,
}

impl<G: GraphType> MatchPredicate<G> {

    pub fn new(criteria: &Predicate<G>) -> Self {
    
        todo!();
        /*
            : criteria_(criteria)
        */
    }
    
    #[inline] pub fn get_criteria(&self) -> Predicate<G> {
        
        todo!();
        /*
            return criteria_;
        */
    }
    
    #[inline] pub fn get_count(&self) -> i32 {
        
        todo!();
        /*
            return count_;
        */
    }
    
    #[inline] pub fn count(&mut self, count: i32) -> &mut MatchPredicate<G> {
        
        todo!();
        /*
            count_ = count;
        return *this;
        */
    }
    
    #[inline] pub fn star_count(&mut self) -> &mut MatchPredicate<G> {
        
        todo!();
        /*
            return count(kStarCount);
        */
    }

    #[inline] pub fn non_terminal(&mut self) -> &mut MatchPredicate<G> {
        
        todo!();
        /*
            nonTerminal_ = true;
        return *this;
        */
    }
    
    #[inline] pub fn exclude_from_subgraph(&mut self) -> &mut MatchPredicate<G> {
        
        todo!();
        /*
            includeInSubgraph_ = false;
        return *this;
        */
    }
    
    #[inline] pub fn is_non_terminal(&self) -> bool {
        
        todo!();
        /*
            return nonTerminal_;
        */
    }
    
    #[inline] pub fn should_include_in_subgraph(&self) -> bool {
        
        todo!();
        /*
            return includeInSubgraph_;
        */
    }
    
    #[inline] pub fn get_debug_string(&self) -> String {
        
        todo!();
        /*
            return debugString_;
        */
    }
    
    #[inline] pub fn set_debug_string(&mut self, debug_string: &String)  {
        
        todo!();
        /*
            debugString_ = debugString;
        */
    }
}
