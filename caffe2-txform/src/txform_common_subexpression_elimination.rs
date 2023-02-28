crate::ix!();

use crate::{
    Graph,
    Transform
};

/**
| Common Subexpression Elimination
|
| This transforms looks for specific operators
| (denoted by allowed_ops_), and removes
|  unnecessary repetition of that operator.
|
| Consider some operator of X, that reads from
| blob b_ written to by W. X_a and X_b read the
|  output of X. However, another operator Y, is
|  the same type as X, has the same arguments as
|  X, and reads from the same input b_, written to
|  by W. It's output is the same as X. Y_a, Y_b,
|  and Y_c read from Y.
|
| Then, we can eliminate the common
| subexpressions X and Y, and merge them to Z,
|  where X_a, X_b, Y_a, Y_b, and Y_c all read from
|  Z.
|
|
| TODO(benz): Fix the error to not match nodes
| that write to external output.
*/
pub struct CommonSubexpressionEliminationTransform {

    allowed_ops: HashSet<String>, // default = {"LearningRate", "FC"};
}

impl Transform for CommonSubexpressionEliminationTransform {

}

impl Default for CommonSubexpressionEliminationTransform {
    
    fn default() -> Self {
        todo!();
        /*
            SetPatternMatchType(SORTED_WRT_EXECUTION_ORDER)
        */
    }
}

impl CommonSubexpressionEliminationTransform {
    
    #[inline] pub fn is_allowed(&mut self, op_type: String) -> bool {
        
        todo!();
        /*
            return allowed_ops_.count(op_type);
        */
    }
    
    #[inline] pub fn pattern_rule(
        &mut self, 
        g: &Graph,
        subgraph: &Vec<i32>,
        idx: i32) -> bool 
    {
        todo!();
        /*
            if (subgraph.size() == 0) {
        if (IsAllowed(g.node(idx).op.type()))
          return true;
        return false;
      }
      return are_nodes_common(g, subgraph.at(0), idx);
        */
    }

    /**
      | As long as we have matched more than 2
      | ops, it is worth eliminating.
      |
      */
    #[inline] pub fn validator_rule(
        &mut self, 
        g: &Graph,
        subgraph: &Vec<i32>) -> bool 
    {
        todo!();
        /*
            if (subgraph.size() >= 2) {
        return true;
      }
      return false;
        */
    }
    
    #[inline] pub fn replace_rule(
        &mut self, 
        subgraph: &Vec<i32>,
        g_ptr: *mut Graph) -> bool 
    {
        todo!();
        /*
            CHECK(g_ptr);
      auto& g = *g_ptr;

      // We're gonna make a new node, with the same input as all of the ones in
      // subgraph, but with their combined children.
      int new_idx = g.size();
      OperatorDef new_op = g.node(subgraph[0]).op;
      // We will need to rename the output blobs.
      new_op.clear_output();
      for (const auto& blob : g.node(subgraph[0]).op.output()) {
        new_op.add_output("transform/" + blob);
      }

      // Need to set up the parents.
      const auto& new_op_parents = g.node(subgraph[0]).parents;

      for (auto& parent : new_op_parents) {
        int parent_idx = parent.first;

        // Make the parents acknowledge us as its new child.
        g.node(parent_idx).children[new_idx] = new_op_parents.at(parent_idx);

        // Make the parents disown all our outdated siblings.
        for (const auto i : c10::irange(subgraph.size())) {
          g.node(parent_idx).children.erase(subgraph[i]);
        }
      }

      // Add the node now.
      g.push_node(
          Node(new_op, true, new_op_parents, std::map<int, std::vector<string>>()));

      // Now, we need to populate the child edges.
      for (const int x : subgraph) {
        // Figure out what the subgraph's node's blobs correspond to in new_op
        // This is easy, since their indices match.
        std::map<string, string> output_renamings;
        for (int i = 0; i < new_op.output_size(); i++) {
          output_renamings[g.node(x).op.output(i)] = g.node(new_idx).op.output(i);
        }

        // Now, time to add the old node's children to new_op
        for (auto& child : g.node(x).children) {
          int child_idx = child.first;
          std::vector<string> blobs = child.second;

          // rename the old blobs, and use them for our new edge.
          for (string& blob : blobs) {
            blob = output_renamings.at(blob);
          }

          // create this new edge
          g.node(new_idx).children[child_idx] = blobs;
          g.node(child_idx).parents[new_idx] = blobs;

          // delete the old edge
          g.node(child_idx).parents.erase(x);

          // need to rename the inputs of the children too.
          for (int i = 0; i < g.node(child_idx).op.input_size(); i++) {
            string blob = g.node(child_idx).op.input(i);
            if (output_renamings.count(blob) > 0) {
              g.node(child_idx).op.set_input(i, output_renamings.at(blob));
            }
          }
        }
      }

      g.DeactivateSubgraph(subgraph);

      return true;
        */
    }
}

register_transform!{
    CommonSubexpressionElimination,
    CommonSubexpressionEliminationTransform
}

/**
  | Checks if the node at model_idx and the node
  | at candidate_idx are "common
  | subexpressions". That is, do they have the
  | same function, and take in the exact same
  | input. If so, then their function is
  | duplicated.
  */
#[inline] pub fn are_nodes_common(
    g:             &Graph,
    model_idx:     i32,
    candidate_idx: i32) -> bool 
{
    todo!();
    /*
        // We need the candidate operator to match this model_op.
      const Node& model_node = g.node(model_idx);
      const Node& candidate_node = g.node(candidate_idx);

      // Types need to match.
      if (model_node.op.type() != candidate_node.op.type()) {
        return false;
      }
      // Arguments need to match.
      if (!MatchArguments(model_node.op, candidate_node.op)) {
        return false;
      }
      // Inputs need to match.
      if (model_node.op.input_size() != candidate_node.op.input_size()) {
        return false;
      }
      // If any input_blob name is different, this is not okay.
      for (int i = 0; i < model_node.op.input_size(); i++) {
        if (candidate_node.op.input(i) != model_node.op.input(i)) {
          return false;
        }
      }
      // Now, we also need to check that each blob comes from the same parent, or
      // if they are external (isn't in parents). This is equivalent to a
      // map equality (since parent edges can only contain up to one blob).
      if (model_node.parents.size() != candidate_node.parents.size() ||
          !std::equal(
              model_node.parents.begin(),
              model_node.parents.end(),
              candidate_node.parents.begin())) {
        return false;
      }

      // Output size have to match too.
      if (model_node.op.output_size() != candidate_node.op.output_size()) {
        return false;
      }
      return true;
    */
}
