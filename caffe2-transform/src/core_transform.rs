crate::ix!();

use crate::{
    PatternMatchType,
    NetDef,
    Graph,
};

/**
  | Determines the type of subgraphs that
  | 
  | PatternMatch will find.
  | 
  | CONNECTED_SUBGRAPH will only match
  | subgraphs that are connected.
  | 
  | These subgraphs satisfy that every
  | node of the match is connected to the
  | subgraph of the nodes that come before
  | it.
  | 
  | For example, in the graph (1) --> (2)
  | --> (3) --> (4),
  | 
  | This is capable of matching the subgraph
  | [2, 3] and [4, 3]
  | 
  | This is not capable of matching the subgraph
  | [2, 4].
  | 
  | SORTED_WRT_EXECUTION_ORDER will
  | match subgraphs that guarantee sorted
  | execution order.
  | 
  | The nodes don't have to be connected.
  | It is faster than General.
  | 
  | For example, in the graph (1) --> (2)
  | --> (3) --> (4),
  | 
  | This is capable of matching the subgraph
  | [2, 4], [3, 4].
  | 
  | This is not capable of matching the subgraph
  | [3, 1], [4, 3].
  | 
  | GENERAL can match any subgraph.
  | 
  | For example, in the graph (1) --> (2)
  | --> (3) --> (4),
  | 
  | This is capable of matching subgraphs
  | [2, 4], [3, 4], [4, 2, 1].
  | 
  | There is no ordered subgraph of G that
  | cannot be matched by this.
  |
  */
pub enum TransformPatternMatchType {
    CONNECTED_SUBGRAPH,
    SORTED_WRT_EXECUTION_ORDER,
    GENERAL
}

/**
  | The Transform Base Object
  | 
  | A Transform is an operation which manipulates
  | a Caffe2 NetDef.
  | 
  | You can consider it as a function:
  | 
  | Transform.ApplyTo(NetDef) -> NetDef
  | 
  | A Transform Operation does 4 things:
  | 
  | 1) Creates a Graph object from a NetDef,
  | which stores connections.
  | 
  | 2) Pattern Matches on the Graph, to
  | find subgraphs it wants to change.
  | 
  | 3) Replaces the subgraphs that it's
  | matched with new operators.
  | 
  | 4) Creates a NetDef from the changed
  | Graph, and returns it.
  | 
  | The effect of a Transform is defined
  | by its 3 protected virtual functions.
  | 
  | 1) PatternRule determines for an ordered
  | subgraph and a node, whether to consider
  | adding the node to the subgraph.
  | 
  | 2) ValidatorRule determines, for
  | an ordered subgraph, whether it is a
  | match.
  | 
  | 3) ReplaceRule mutates the graph,
  | based on a matched subgraph.
  | 
  | This is the base class for all derived
  | classes to base off. To create your own
  | transform, write your implementations
  | for PatternRule, ValidatorRule, and
  | 
  | ReplaceRule.
  |
  */

/*
 | pub struct Transform {
 |     pattern_match_type: PatternMatchType, // default = CONNECTED_SUBGRAPH
 | }
 */

#[macro_export] macro_rules! REGISTER_TRANSFORM {
    ($name:ident, $($arg:ident),*) => {
        todo!();
        /*
        C10_REGISTER_CLASS(TransformRegistry, name, __VA_ARGS__)
        */
    }
}

pub trait Transform {
    
    /**
      | Generates all matches (stored as ordered
      | subgraphs) and returns them.
      | 
      | A match is stored as vector<int>, which
      | is a mapping to OperatorDefs in Graph.
      | The order matters.
      |
      */
    #[inline] fn pattern_match(&mut self, graph: &Graph) -> Vec<Vec<i32>> {
        
        todo!();
        /*
            // checks if the node at index i is matched already or not
      std::vector<bool> matched(graph.size(), false);

      // stores matches, which are ordered subgraphs of G
      std::vector<std::vector<int>> matches;

      // Consider every possible node as the starting point.
      for (int idx = 0; idx < (int)graph.size(); ++idx) {
        // The current working subgraph. We will try to add new nodes to this,
        // when invoking the PatternRule.
        std::vector<int> subgraph;

        // The largest "validated" subgraph found so far.
        // This will be mutated by PatternMatchHelper.
        std::vector<int> best_subgraph;

        // Only begin to match if the start node is accepted.
        if (!matched.at(idx) && PatternRule(graph, subgraph, idx)) {
          subgraph.push_back(idx);
          PatternMatchHelper(graph, matched, &subgraph, &best_subgraph);
          subgraph.pop_back();
        }
        if (best_subgraph.size() > 0) { // match found
          matches.push_back(best_subgraph);
          for (const auto& x : best_subgraph) {
            matched[x] = true;
          }
        }
      }
      return matches;
        */
    }
    
    /**
      | Attempts to append each neighbor to
      | the end of the subgraph.
      |
      */
    #[inline] fn try_neighbors(
        &mut self, 
        graph:                 &Graph,
        neighbors:             &HashMap<i32,Vec<String>>,
        matched:               &Vec<bool>,
        subgraph_ptr:          *mut Vec<i32>,
        best_subgraph_ptr:     *mut Vec<i32>)  
    {
        todo!();
        /*
            auto& subgraph = *subgraph_ptr;
      for (const auto& edge : neighbors) {
        int j = edge.first;
        if (std::find(subgraph.begin(), subgraph.end(), j) == subgraph.end()) {
          if (!matched.at(j) && PatternRule(graph, subgraph, j)) {
            subgraph.push_back(j);
            PatternMatchHelper(graph, matched, subgraph_ptr, best_subgraph_ptr);
            subgraph.pop_back();
          }
        }
      }
        */
    }
    
    /**
      | A helper function for PatternMatch,
      | which keeps track of the best subgraph
      | so far.
      |
      */
    #[inline] fn pattern_match_helper(
        &mut self, 
        graph:             &Graph,
        matched:           &Vec<bool>,
        subgraph_ptr:      *mut Vec<i32>,
        best_subgraph_ptr: *mut Vec<i32>)  
    {

        todo!();
        /*
            CHECK(subgraph_ptr);
      auto& subgraph = *subgraph_ptr;
      CHECK(best_subgraph_ptr);
      auto& best_subgraph = *best_subgraph_ptr;

      // If the current subgraph is valid, and the largest we've seen so far,
      // make it the best_subgraph.
      if (ValidatorRule(graph, subgraph) &&
          subgraph.size() > best_subgraph.size()) {
        best_subgraph = subgraph;
      }

      size_t size_before = subgraph.size();

      if (pattern_match_type_ == CONNECTED_SUBGRAPH) {
        // Connected Component Order Pattern Matching
        // We want to match subgraphs which are connected ConnectedComponents

        // Try adding each parent and child of every node in the subgraph,
        // and see if we can accept it.
        for (size_t i = 0; i < subgraph.size(); i++) {
          int x = subgraph[i];
          TryNeighbors(
              graph,
              graph.node(x).children,
              matched,
              subgraph_ptr,
              best_subgraph_ptr);
          CAFFE_ENFORCE(
              size_before == subgraph.size(),
              "Subgraph size should not change after returning from recursive call.");
          TryNeighbors(
              graph,
              graph.node(x).parents,
              matched,
              subgraph_ptr,
              best_subgraph_ptr);
          CAFFE_ENFORCE(
              size_before == subgraph.size(),
              "Subgraph size should not change after returning from recursive call.");
        }
      } else if (pattern_match_type_ == SORTED_WRT_EXECUTION_ORDER) {
        // Sorted Execution Order Pattern matching
        // We want to be able to match subgraphs in sorted execution order

        // We can safely assume our subgraph is already sorted.
        // This means, we only need to consider nodes that come after the LAST
        // node in our current subgraph.
        // Thus, we simply iterate over the nodes that come AFTER the last node of
        // our current subgraph.
        size_t start_idx = 0;
        if (subgraph.size() > 0) {
          start_idx = subgraph.back() + 1;
        }
        for (size_t i = start_idx; i < graph.size(); i++) {
          if (!matched.at(i) && PatternRule(graph, subgraph, i)) {
            subgraph.push_back(i);
            PatternMatchHelper(graph, matched, subgraph_ptr, best_subgraph_ptr);
            subgraph.pop_back();
          }
        }
      } else if (pattern_match_type_ == GENERAL) {
        // General Pattern matching
        // We want to be able to match any ordered subgraph

        // For every current subgraph, we consider all nodes to be
        // the next candidate node, as long as it isn't already matched.
        for (size_t i = 0; i < graph.size(); i++) {
          if (std::find(subgraph.begin(), subgraph.end(), i) == subgraph.end()) {
            // Then we try appending it to the subgraph.
            if (!matched.at(i) && PatternRule(graph, subgraph, i)) {
              subgraph.push_back(i);
              PatternMatchHelper(graph, matched, subgraph_ptr, best_subgraph_ptr);
              subgraph.pop_back();
            }
          }
        }
      } else {
        CAFFE_NOT_IMPLEMENTED;
      }
        */
    }
    
    /**
      | The PatternRule essentially answers:
      | 
      | Given the current subgraph (ordered),
      | should we append the new node at idx?
      |
      */
    #[inline] fn pattern_rule(
        &mut self, 
        g:        &Graph,
        subgraph: &Vec<i32>,
        idx:      i32) -> bool 
    {

        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }
    
    /**
      | The ValidatorRule essentially answers:
      | 
      | Given a subgraph, can we accept it?
      |
      */
    #[inline] fn validator_rule(
        &mut self, 
        g: &Graph,
        subgraph: &Vec<i32>) -> bool {

        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }
    
    /**
      | The ReplaceRule actually mutates the
      | graph, and applies the transformation
      | upon the subgraph.
      |
      */
    #[inline] fn replace_rule(
        &mut self, 
        subgraph: &Vec<i32>,
        g_ptr: *mut Graph) -> bool 
    {
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }
    
    #[inline] fn set_pattern_match_type(&mut self, ty: PatternMatchType)  {
        
        todo!();
        /*
            pattern_match_type_ = type;
        */
    }
    
    /**
      | Applies the replace rule onto each of
      | the matches found.
      |
      */
    #[inline] fn replace_pattern(
        &mut self, 
        matches: &Vec<Vec<i32>>,
        graph:   *mut Graph)  
    {
        
        todo!();
        /*
            for (const auto& match : matches) {
        // Make sure each matched node is still active (not overwritten)
        bool is_match_active = true;
        for (int idx : match) {
          if (!graph->is_node_active(idx)) {
            is_match_active = false;
          }
        }

        // Simply try to apply the replace rule upon every match.
        if (is_match_active && !ReplaceRule(match, graph)) {
          CAFFE_THROW("Replace failed!");
        }
      }
        */
    }

    /**
      | Apply a Transform onto a NetDef.
      | 
      | Returns the transformed NetDef.
      | 
      | The simple interface - performs the
      | transformation upon a NetDef, and returns
      | the result.
      |
      */
    #[inline] fn apply_to(&mut self, orig_net: &NetDef) -> NetDef {
        
        todo!();
        /*
            Graph g(orig_net);
      const auto matches = PatternMatch(g);
      ReplacePattern(matches, &g);
      return g.GetNetDef();
        */
    }
}

/**
  | Creates a Transform based on a key, which
  | should be defined in registry.
  |
  */
#[inline] pub fn create_transform(key: String) -> Box<dyn Transform> {
    
    todo!();
    /*
        auto t = TransformRegistry()->Create(key);
      CAFFE_ENFORCE(t != nullptr, "Transform not found in registry: ", key);
      return t;
    */
}

/**
  | Create a Transform object from registry,
  | and immediately apply it to a Netdef.
  |
  */
#[inline] pub fn apply_transform(
    key: &String,
    netdef: &NetDef) -> NetDef 
{
    todo!();
    /*
        auto t = CreateTransform(key);
      return t->ApplyTo(netdef);
    */
}

#[inline] pub fn average_net_run_duration(
    netdef:      &NetDef,
    init_netdef: &NetDef,
    warmup_runs: i32,
    main_runs:   i32) -> f64 
{

    todo!();
    /*
        Workspace ws;
      if (init_netdef.op_size() > 0) {
        std::unique_ptr<NetBase> init_net(CreateNet(init_netdef, &ws));
        CHECK(init_net);
        CAFFE_ENFORCE(init_net->Run(), "Init run has failed!");
      } else {
        // If a proper init_net is not provided, then this is the best we can do.
        for (auto inp : netdef.external_input()) {
          ws.CreateBlob(inp);
        }
      }
      std::unique_ptr<NetBase> net(CreateNet(netdef, &ws));
      CHECK(net);
      CAFFE_ENFORCE(
          warmup_runs >= 0,
          "Number of warm up runs should be non negative, provided ",
          warmup_runs,
          ".");

      for (int i = 0; i < warmup_runs; i++) {
        CAFFE_ENFORCE(net->Run(), "Warmup run ", i, " has failed.");
      }

      CAFFE_ENFORCE(
          main_runs > 0,
          "Number of main runs should be positive, provided ",
          main_runs,
          ".");
      Timer timer;
      for (int i = 0; i < main_runs; i++) {
        CAFFE_ENFORCE(net->Run(), "Main run ", i, " has failed.");
      }
      return timer.MilliSeconds();
    */
}

/**
  | Create a Transform object from registry,
  | apply it to a NetDef.
  | 
  | Will only return the transformed net
  | if it is faster than the old net.
  | 
  | This will run the init net first, will
  | run the two nets warmup_runs times.
  | 
  | Then, we will take the average time of
  | main_runs runs, and only keep the transformed
  | net if it is faster by a factor of improvement_threshold.
  |
  */
#[inline] pub fn apply_transform_if_faster(
    key:                   &String,
    netdef:                &NetDef,
    init_netdef:           &NetDef,
    warmup_runs:           i32,
    main_runs:             i32,
    improvement_threshold: f64) -> NetDef {

    todo!();
    /*
        NetDef transformed_netdef = ApplyTransform(key, netdef);
      double original_net_time =
          average_net_run_duration(netdef, init_netdef, warmup_runs, main_runs);
      double new_net_time = average_net_run_duration(
          transformed_netdef, init_netdef, warmup_runs, main_runs);
      if (original_net_time > improvement_threshold * new_net_time) {
        return transformed_netdef;
      }
      return netdef;
    */
}
