crate::ix!();

pub type Criteria           = String;
pub type TestMatchGraph     = MatchGraph<NNGraph>;
pub type TestMatchPredicate = MatchPredicate<NNGraph>;

/**
  | Each match is a struct of subgraph and
  | map from the string used in the query
  | to a NodeRef in the subgraph note:
  | 
  | the maps are injective but not necessarily
  | bijective -- if you use the same name
  | in the query twice only one will be mapped.
  | 
  | See `getMatches` to generate these
  | structs.
  |
  */
pub struct MatchedSubgraph<T,U> {

    /**
      | A subgraph that contains at least all the
      | nodes in matchMap
      |
      | This is the canonical match -- the
      | matchMap is only a useful utility
      */
    subgraph:  SubgraphType<T,U>,

    /**
      | Maps a variable name to a Node in a dataflow
      | graph
      |
      */
    match_map:  HashMap<String,NodeRef<T,U>>,
}

/**
  | Provides safer access to matchMap with
  | nicer semantics
  |
  */
impl<T,U> Index<String> for MatchedSubgraph<T,U> {

    type Output = NodeRef<T,U>;

    fn index(&self, key: String) -> &Self::Output {
        todo!();
        /*
          auto search = matchMap.find(key);
          CAFFE_ENFORCE(
              search != matchMap.end(), "Could not find key in map of matches:", key);
          return search->second;
        */
    }
}

