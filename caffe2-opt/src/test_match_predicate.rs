crate::ix!();

lazy_static!{
    static ref mtx: parking_lot::RawMutex = todo!();
}

#[inline] pub fn dealloc_token_strings()  {
    
    todo!();
    /*
        for (auto p : tokens) {
        delete (std::string*)p;
      }
      tokens.clear();

      for (auto p : tokenVectors) {
        delete (std::vector<void*>*)p;
      }
      tokenVectors.clear();
    */
}

/**
  | Node matches a criteria (string) if
  | the data string is the same as the criteria.
  | Special case: "*" will match any thing.
  |
  */
#[inline] pub fn test_match_predicate(criteria: &Criteria) -> TestMatchPredicate {
    
    todo!();
    /*
        auto predicate =
          TestMatchPredicate([criteria](nom::repr::NNGraph_NodeRef nodeRef) {
            std::string nodeLabel = getNodeName(nodeRef);
            return (criteria == "*" || criteria == nodeLabel);
          });
      predicate.setDebugString(criteria);
      return predicate;
    */
}

