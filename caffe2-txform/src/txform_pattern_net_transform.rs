crate::ix!();


/**
  | PatternNetTransform allows you to
  | create transforms using a simple interface.
  | 
  | Simply provide a Pattern NetDef and
  | a Replace
  | 
  | NetDef, and this Transform will find
  | subgraphs which fit the pattern net,
  | and replace it with the replace net.
  |
  */
pub struct PatternNetTransform {

    /// Graph of Pattern NetDef
    p: Graph,

    /**
      | The Traversal Order of the Pattern Net's
      | Operators
      |
      | This is a permutation of the numbers from
      | {0, ..., p.size()-1}
      */
    ordered_ops: Vec<i32>,

    /**
      | The Inverse of the Traversal Order of the
      | Pattern Net's Operators
      |
      | That is, inverse_ops[ordered_ops[i]] ==
      | i is always true.
      */
    inverse_ops: Vec<i32>,

    /// Graph of Replace NetDef
    r: Graph,

    /**
      | This flag determines if the transform
      | will match operator arguments.
      |
      */
    argument_match: bool, // default = false

    ssa_id: i32, // default = 0
}

impl Transform for PatternNetTransform {

}


impl PatternNetTransform {

    pub fn new(pattern_net: &NetDef, replace_net: &NetDef) -> Self {
        todo!();
        /*
            : p_(transform::Graph(pattern_net)), r_(transform::Graph(replace_net)) 

        // external input and output must match!
        CAFFE_ENFORCE(
            p_.external_input() == r_.external_input(),
            "External inputs do not match!");
        CAFFE_ENFORCE(
            p_.external_output() == r_.external_output(),
            "External outputs do not match!");
        ordered_ops_ = GetPatternTraversalOrder(p_);
        inverse_ops_.resize(ordered_ops_.size());
        for (size_t i = 0; i < ordered_ops_.size(); i++) {
          inverse_ops_[ordered_ops_[i]] = i;
        }
        */
    }
    
    #[inline] pub fn enable_argument_matching(&mut self)  {
        
        todo!();
        /*
            argument_match_ = true;
        */
    }
    
    #[inline] pub fn disable_argument_matching(&mut self)  {
        
        todo!();
        /*
            argument_match_ = false;
        */
    }
    
    #[inline] pub fn transform_blob_wrapper(&mut self, blob_name: &String) -> String {
        
        todo!();
        /*
            return "transform/" + blob_name + "_" + c10::to_string(ssa_id_);
        */
    }

    /**
      This returns a permutation of the Pattern Net's operators.
      The permutation satisfies this property:
      - For any index i, order(i) is a neighbor of some node from
      {order(1), ..., order(i-1)}.

      Why is this important? Consider the following case:
      PatternNet: 0 ---> 2 <--- 1

      When we have matched onto [0], and trying to add [1] to our subgraph,
      we cannot, since PatternMatch only considers neighbors of the current
      subgraph as a candidate next node.

      Therefore, we must present the subgraph in an order such that each node is
      a neighbor of its prefix subgraph. One ordering for the above example is
      [0, 2, 1].

      First, single source traverse through the netdef.
      This ensures all newly ordered are reachable from their prefix subset
      Outputs a permutation of the operators.
      */
    #[inline] pub fn get_pattern_traversal_order(&mut self, graph: &Graph) -> Vec<i32> {
        
        todo!();
        /*
            std::vector<bool> visited(graph.size(), false);
      std::vector<int> ordered_ops;
      std::queue<int> q;
      if (graph.size() > 0) {
        q.push(0);
        ordered_ops.push_back(0);
        visited[0] = true;
      }
      while (!q.empty()) {
        int idx = q.front();
        q.pop();
        for (const auto& edge : graph.node(idx).children) {
          int x = edge.first;
          if (!visited[x]) {
            q.push(x);
            ordered_ops.push_back(x);
            visited[x] = true;
          }
        }
        for (const auto& edge : graph.node(idx).parents) {
          int x = edge.first;
          if (!visited[x]) {
            q.push(x);
            ordered_ops.push_back(x);
            visited[x] = true;
          }
        }
      }
      CAFFE_ENFORCE(
          ordered_ops.size() == graph.size(), "Pattern graph must be connected.");
      return ordered_ops;
        */
    }

    /**
      | We want to the final result of subgraph
      | to match the PatternNet in the order
      | of ordered_ops, operator by operator.
      | 
      | [[[ ie. g.node(subgraph[i]) should
      | match p.node(ordered_ops[i]) ]]]
      | 
      | PatternRule for PatternNetTransform
      | does the following:
      | 
      | When trying to insert node idx into subgraph[p_idx],
      | we need to see if the edges between index
      | and the subgraph match the edges between
      | p[ordered_ops[idx]] and p[ordered_ops[0]...ordered_ops[p_idx-1]].
      | 
      | g.node(subgraph[i]) should match
      | p_.node(ordered_ops_[i])
      | 
      | g.node(g_idx) should match p_.node(p_idx)
      |
      */
    #[inline] pub fn pattern_rule(
        &mut self, 
        g:        &Graph,
        subgraph: &Vec<i32>,
        g_idx:    i32) -> bool 
    {
        
        todo!();
        /*
            if (subgraph.size() >= ordered_ops_.size()) {
        return false;
      }
      int p_idx = ordered_ops_[subgraph.size()];

      if (!compare_ops(p_.node(p_idx).op, g.node(g_idx).op, argument_match_)) {
        return false;
      }

      // Let's say ordered_ops_ is [0, 2, 1], with 0 -> 2 being an edge
      // When we try to match onto the second element, let's say our
      // subgraph so far is [4], with it trying to become [4, 5].
      // Then, we need to show that since 0 -> 2 is an edge is ordered_ops_,
      // 4 must be a direct parent of 5 in the subgraph
      // (the indices must match).
      // Similarly, assume there is an edge from 1 -> 2 in p_.
      // When trying to match [4, 5] to [4, 5, 7], we must verify that
      // there exists an edge from 7 -> 5 in G.
      for (const auto& edge : p_.node(p_idx).parents) {
        int parent = edge.first;
        // g_idx doesn't have parent in subgraph that p_[p_idx] has
        // inverse_ops_ gets the index of a p_idx inside of ordered_ops_.
        if (inverse_ops_[parent] < subgraph.size() &&
            g.node(g_idx).parents.count(subgraph[inverse_ops_[parent]]) == 0) {
          return false;
        }
      }

      for (const auto& edge : p_.node(p_idx).children) {
        int child = edge.first;
        if (inverse_ops_[child] < subgraph.size() &&
            g.node(g_idx).children.count(subgraph[inverse_ops_[child]]) == 0) {
          return false;
        }
      }
      return true;
        */
    }

    /**
      | ValidatorRule for PatternNetTransform
      | does the following:
      | 
      | Checks if the size of subgraph and p.size()
      | are the same. That's it!
      |
      */
    #[inline] pub fn validator_rule(
        &mut self, 
        g: &Graph,
        subgraph: &Vec<i32>) -> bool 
    {
        todo!();
        /*
            // Due to strict PatternRule, it suffices to simply check for size
      return subgraph.size() == p_.size();
        */
    }
    
    /**
      | ReplaceRule for PatternNet Transform
      | does the following:
      | 
      | 1) Figure out edge renamings for edges
      | going into/out of the subgraph.
      | 
      | That is, for each blob in the pattern
      | graph, what is it called in the matched
      | subgraph?
      | 
      | 2) Remove the matched subgraph.
      | 
      | 3) Append the replace graph's operators
      | to the graph's operators, and use the
      | renamings to rename the blob names.
      | 
      | 4) Create all the children/parent
      | relationships within the replaced
      | graph, and stitch together the inputs
      | and outputs into the rest of the graph,
      | matching the removed subgraph.
      |
      */
    #[inline] pub fn replace_rule(
        &mut self, 
        match_: &Vec<i32>,
        g_ptr: *mut Graph) -> bool {
        
        todo!();
        /*
            CHECK(g_ptr);
      auto& g = *g_ptr;

      ssa_id_++;

      // Map of PatternNet blob name to Matched blob name.
      // Figures out how to rename the pattern_net to make the replacement fit.
      std::unordered_map<string, string> external_renaming;

      // Figure out blob renamings
      for (const auto i : c10::irange(match.size())) {
        int g_idx = match[i];
        int p_idx = ordered_ops_[i];
        for (int j = 0; j < p_.node(p_idx).op.input().size(); j++) {
          string p_blob = p_.node(p_idx).op.input(j);
          string g_blob = g.node(g_idx).op.input(j);
          if (p_.external_input().count(p_blob)) {
            external_renaming[p_blob] = g_blob;
          }
        }
        for (int j = 0; j < p_.node(p_idx).op.output().size(); j++) {
          string p_blob = p_.node(p_idx).op.output(j);
          string g_blob = g.node(g_idx).op.output(j);
          if (p_.external_output().count(p_blob)) {
            external_renaming[p_blob] = g_blob;
          }
        }
      }

      auto input_list = g.GetSubgraphInput(match);
      auto output_list = g.GetSubgraphOutput(match);

      g.DeactivateSubgraph(match);

      int offset = g.size();

      g.resize_nodes(offset + r_.size());

      // Append all the new operators.
      for (const auto i : c10::irange(r_.size())) {
        int new_node_idx = offset + i;

        OperatorDef new_op = r_.node(i).op;

        new_op.clear_input();
        new_op.clear_output();
        // Stitch Input from external graph into replaced subgraph
        for (const auto& blob : r_.node(i).op.input()) {
          if (external_renaming.count(blob)) {
            string new_blob = external_renaming[blob];
            new_op.add_input(new_blob);

            // binary searches for new_blob amongst input list.
            auto it = std::lower_bound(
                input_list.begin(), input_list.end(), std::make_pair(new_blob, -1));

            // if the input came from the graph (instead of G's external input)
            for (; it < input_list.end() && it->first == new_blob; it++) {
              int parent = it->second;
              g.node(parent).children[new_node_idx].push_back(new_blob);
              g.node(new_node_idx).parents[parent].push_back(new_blob);
            }
          } else {
            new_op.add_input(TransformBlobWrapper(blob));
          }
        }
        // Stitch Output from replaced subgraph to external graph.
        for (const auto& blob : r_.node(i).op.output()) {
          if (external_renaming.count(blob)) {
            string new_blob = external_renaming[blob];
            new_op.add_output(new_blob);

            // binary searches for new_blob amongst input list.
            auto it = std::lower_bound(
                output_list.begin(),
                output_list.end(),
                std::make_pair(new_blob, -1));

            // if the output goes to the graph (instead of G's external output)
            for (; it < output_list.end() && it->first == new_blob; it++) {
              int child = it->second;
              g.node(child).parents[new_node_idx].push_back(new_blob);
              g.node(new_node_idx).children[child].push_back(new_blob);
            }
          } else {
            new_op.add_output(TransformBlobWrapper(blob));
          }
        }

        // Connect all internal edges within replace graph
        for (const auto& edge : r_.node(i).parents) {
          int parent = edge.first;
          int new_node_parent = offset + parent;
          const auto& blobs = edge.second;
          for (const string& blob : blobs) {
            g.node(new_node_idx)
                .parents[new_node_parent]
                .push_back(TransformBlobWrapper(blob));
          }
        }

        for (const auto& edge : r_.node(i).children) {
          int child = edge.first;
          int new_node_child = offset + child;
          const auto& blobs = edge.second;
          for (const string& blob : blobs) {
            g.node(offset + i)
                .children[new_node_child]
                .push_back(TransformBlobWrapper(blob));
          }
        }

        g.node(new_node_idx).op = new_op;
        g.node(new_node_idx).active = true;
      }
      return true;
        */
    }
}

#[inline] pub fn compare_ops(
    p_op:      &OperatorDef,
    g_op:      &OperatorDef,
    arg_match: bool) -> bool 
{
    
    todo!();
    /*
        // must specify a type for pattern operators
      CAFFE_ENFORCE(
          p_op.has_type(), "Types must be specified for all pattern operators.");
      if (!MatchStrings(p_op.type(), g_op.type())) {
        return false;
      }
      // ensure number of inputs are the same
      if (p_op.input().size() != g_op.input().size()) {
        return false;
      }

      // ensure number of outputs are the same
      if (p_op.output().size() != g_op.output().size()) {
        return false;
      }

      if (p_op.has_device_option()) {
        if (!g_op.has_device_option() ||
            p_op.device_option().device_type() !=
                g_op.device_option().device_type()) {
          return false;
        }
      }

      // make sure engine is the same (if specified in pattern)
      if (p_op.has_engine() && !MatchStrings(p_op.engine(), g_op.engine())) {
        return false;
      }
      // If argument_match is specified, make sure those are the same.
      if (arg_match) {
        if (!MatchArguments(p_op, g_op)) {
          return false;
        }
      }
      return true;
    */
}
