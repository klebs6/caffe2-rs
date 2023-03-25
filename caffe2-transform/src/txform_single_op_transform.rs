crate::ix!();


/**
  | Single Op Transform Base class
  | 
  | A transform which is applied to a single
  | node, in place.
  | 
  | Transforms which derive from SingleOpTransform
  | need to override:
  | 
  | ReplaceOperator and MatchOperator.
  |
  */
pub trait SingleOpTransform: Transform {

    /**
      | Specify what the op needs to be to match
      | the pattern.
      |
      */
    fn match_operator(&mut self, op: &OperatorDef);

    /**
      | Specify how the operator should be replaced.
      |
      */
    fn replace_operator(&mut self, op: *mut OperatorDef);
    
    #[inline] fn pattern_rule(
        &mut self, 
        g:        &Graph,
        subgraph: &Vec<i32>,
        idx:      i32) -> bool 
    {
        todo!();
        /*
            if (subgraph.size() == 0) {
        return MatchOperator(g.node(idx).op);
      }
      return false;
        */
    }
    
    #[inline] fn validator_rule(
        &mut self, 
        g:        &Graph,
        subgraph: &Vec<i32>) -> bool 
    {
        todo!();
        /*
            if (subgraph.size() == 1) {
        return true;
      }
      return false;
        */
    }
    
    #[inline] fn replace_rule(
        &mut self, 
        subgraph: &Vec<i32>,
        g_ptr:    *mut Graph) -> bool 
    {
        todo!();
        /*
            CHECK(g_ptr);
      auto& g = *g_ptr;
      ReplaceOperator(&(g.node(subgraph[0]).op));
      return true;
        */
    }
}
