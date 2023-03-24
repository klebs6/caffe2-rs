crate::ix!();

/**
  | Instead of breaking down the DAG into
  | chains, we partition it into clusters
  | of sync ops and individual async op.
  | 
  | This is useful for disturbuted inference
  | case where we have sync and async cpu
  | ops.
  | 
  | -----------
  | @note
  | 
  | we have go sync each aysnc op instead
  | of put them into the chain and sync its
  | tail like GPU op, because CPU async ops
  | are typically rpc calls and are not guaranteed
  | to be linearized at remote site.
  | 
  | Here chains are essentially groups,
  | we used chain/group interchangeably
  |
  */
#[inline] pub fn compute_groups(orig_nodes: &mut Vec<OperatorNode>) -> ExecutionChains {
    
    todo!();
    /*
        const std::vector<OpGraphNode> nodes = pruneOpNodeGraph(orig_nodes);
      ExecutionChains chains;
      std::vector<int> sync_frontier;
      std::vector<int> async_frontier;

      std::vector<int> in_degrees;
      in_degrees.reserve(nodes.size());
      std::transform(
          nodes.begin(),
          nodes.end(),
          std::back_inserter(in_degrees),
          [](const OpGraphNode& n) { return n.parents_.size(); });

      // Screen out the primary root nodes
      for (int idx = 0; idx < (int)nodes.size(); ++idx) {
        if (in_degrees[idx] == 0) {
          if (orig_nodes[idx].operator_->HasAsyncPart()) {
            async_frontier.push_back(idx);
          } else {
            sync_frontier.push_back(idx);
          }
        }
      }

      // We check sync ops on the frontier first and then async ops. This gives us a
      // head start to execute sync ops locally while waiting for async ops to
      // finish.
      std::queue<int> q;
      while (!(async_frontier.empty() && sync_frontier.empty())) {
        // Sync ops
        for (const auto i : sync_frontier) {
          q.push(i);
        }
        sync_frontier.clear();
        std::vector<int> chain;
        while (!q.empty()) {
          int idx = q.front();
          q.pop();
          chain.push_back(idx);
          for (int child : nodes[idx].children_) {
            if (--in_degrees[child] == 0) {
              if (orig_nodes[child].operator_->HasAsyncPart()) {
                async_frontier.push_back(child);
              } else {
                q.push(child);
              }
            }
          }
        }
        // add the whole group of continuous sync ops into one chain
        if (!chain.empty()) {
          chains.emplace(chain.front(), chain);
        }

        // Async ops
        for (const auto i : async_frontier) {
          q.push(i);
        }
        async_frontier.clear();
        while (!q.empty()) {
          int idx = q.front();
          q.pop();
          // Put each individual node as a new chain
          chains[idx] = {idx};
          for (int child : nodes[idx].children_) {
            if (--in_degrees[child] == 0) {
              if (orig_nodes[child].operator_->HasAsyncPart()) {
                q.push(child);
              } else {
                sync_frontier.push_back(child);
              }
            }
          }
        }
      }

      updateOperatorNodes(orig_nodes, chains);
      return chains;
    */
}
