crate::ix!();

/**
  | \brief Main graph matcher interface.
  |
  | This class solves a problem of finding
  | a matching subgraph, which is specified in
  | a text form.
  */
pub struct GraphMatcher {

    match_map:              HashMap<String,NNGraph_NodeRef>,
    var_map:                HashMap<String,TestMatchGraph_NodeRef>,
    call_map:               HashMap<String,TestMatchGraph_NodeRef>,
    match_graph:            TestMatchGraph,
    match_graph_root_node:  TestMatchGraph_NodeRef,
    syntax_is_valid:        bool, // default = true
}

impl GraphMatcher {

    /**
      | @brief
      | 
      | Initialize subgraph pattern from \p
      | 
      | STR.
      |
      */
    #[inline] pub fn init_from_string(&mut self, str: *const u8)  {
        
        todo!();
        /*
            genMatcherFromIRStr(str);
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Initialize subgraph patter from IR
      | stored in file \p fname.
      |
      */
    #[inline] pub fn init_from_file(&mut self, fname: *const u8)  {
        
        todo!();
        /*
            genMatcherFromIRFile(fname);
        */
    }

    /**
      | @brief
      | 
      | Try to find the pattern in the given graph
      | \p DF and return true if it was found.
      |
      */
    #[inline] pub fn find_subgraph(&mut self, df: &mut NNGraph) -> bool {
        
        todo!();
        /*
            return doesMatch(df);
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Replace the found subgraph with another
      | one.
      |
      */
    #[inline] pub fn replace_subgraph_with(&mut self)  {
        
        todo!();
        /*
            CAFFE_THROW("Subgraph replacement is not implemented yet.");
        */
    }

    /// \brief Return the matcher graph.
    #[inline] pub fn get_matcher_graph(&mut self) -> *mut TestMatchGraph {
        
        todo!();
        /*
            return &matchGraph_;
        */
    }
    
    /**
      | TODO: Do we need this, or can we get it
      | from getMatcherGraph?
      |
      */
    #[inline] pub fn get_matcher(&mut self) -> TestMatchGraph_NodeRef {
        
        todo!();
        /*
            return matchGraphRootNode_;
        */
    }
    
    /**
      | @brief
      | 
      | Return a mapping from IR variable name
      | (std::string) to Node in the matched
      | graph.
      |
      */
    #[inline] pub fn get_match_map(&self) -> HashMap<String,NNGraph_NodeRef> {
        
        todo!();
        /*
            return matchMap_;
        */
    }
    
    #[inline] pub fn does_match(&mut self, df: &mut NNGraph) -> bool {
        
        todo!();
        /*
            if (!syntaxIsValid_) {
          return false;
        }
        matchMap_.clear();
        std::vector<nom::repr::NNGraph_NodeRef> Nodes = df.getMutableNodes();
        for (auto& Node : Nodes) {
          auto match =
              matchGraph_.isSubgraphMatch(Node, matchGraphRootNode_, true, true);
          if (match.isMatch()) {
            // Fill the match map
            auto subgraphMatcherMap = match.getMatchNodeMap();
            for (auto p : varMap_) {
              auto iter = subgraphMatcherMap->find(p.second);
              if (iter != subgraphMatcherMap->end()) {
                matchMap_[p.first] = iter->second;
              }
            }
            for (auto p : callMap_) {
              auto iter = subgraphMatcherMap->find(p.second);
              if (iter != subgraphMatcherMap->end()) {
                matchMap_[p.first] = iter->second;
              }
            }

            return true;
          }
        }
        return false;
        */
    }
    
    #[inline] pub fn gen_matcher_from_astexpr(
        &mut self, 
        expr: *mut ASTExpr, 
        insert_temp: Option<bool>) -> TestMatchGraph_NodeRef 
    {
        let insert_temp: bool = insert_temp.unwrap_or(false);

        todo!();
        /*
            if (!expr->isCall()) {
        if (expr->starInputs()) {
          return matchGraph_.createNode(std::move(
              testMatchPredicate(Criteria("*")).starCount().nonTerminal()));
        }
        if (!varMap_.count(expr->name)) {
          varMap_[expr->name] = matchGraph_.createNode(
              std::move(testMatchPredicate(Criteria("*")).nonTerminal()));
        }
        return varMap_[expr->name];
      }

      std::vector<TestMatchGraph_NodeRef> children;
      for (auto child : expr->children) {
        children.push_back(genMatcherFromASTExpr(child, true));
      }

      auto res = matchGraph_.createNode(testMatchPredicate(Criteria(expr->name)));
      callMap_[expr->name] = res;
      for (auto child : children) {
        matchGraph_.createEdge(child, res);
      }

      if (insertTemp) {
        auto temp = matchGraph_.createNode(testMatchPredicate(Criteria("*")));
        matchGraph_.createEdge(res, temp);
        res = temp;
      }

      return res;
        */
    }
    
    #[inline] pub fn gen_matcher_from_aststmt(&mut self, stmt: *mut ASTStmt) -> TestMatchGraph_NodeRef {
        
        todo!();
        /*
            auto right = genMatcherFromASTExpr(stmt->rhs);
      auto res = right;
      /* For cases like
       %x, %y = Foo(%z)
       for now we just say that both %x and %y are defined by node Foo, we don't
       distinguish them (i.e. we don't keep any information about their order. */
      for (auto v : stmt->lhs) {
        res = matchGraph_.createNode(testMatchPredicate(Criteria("*")));
        matchGraph_.createEdge(right, res);
        varMap_[v] = res;
      }
      return res;
        */
    }
    
    #[inline] pub fn gen_matcher_from_astgraph(&mut self, ast: *mut ASTGraph) -> TestMatchGraph_NodeRef {
        
        todo!();
        /*
            matchGraph_ = TestMatchGraph();
      // TODO: Cleanup this.
      TestMatchGraph_NodeRef last = nullptr;
      if (ast->stmts.empty()) {
        syntaxIsValid_ = false; // Temporary solution, which works because we don't
                                // allow empty graphs.
      }

      for (auto stmt : ast->stmts) {
        auto r = genMatcherFromASTStmt(stmt);
        if (r) {
          last = r;
        }
      }

      return last;
        */
    }
    
    #[inline] pub fn gen_matcher_from_irfile(&mut self, fname: *const u8) -> TestMatchGraph_NodeRef {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lock(mtx_);
      ASTGraph g;
      parseFile(fname, &g);
      matchGraphRootNode_ = genMatcherFromASTGraph(&g);
      deallocTokenStrings();
      return matchGraphRootNode_;
        */
    }
    
    #[inline] pub fn gen_matcher_from_irstr(&mut self, str: *const u8) -> TestMatchGraph_NodeRef {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lock(mtx_);
      ASTGraph g;
      parseString(str, &g);
      matchGraphRootNode_ = genMatcherFromASTGraph(&g);
      deallocTokenStrings();
      return matchGraphRootNode_;
        */
    }
    
    /// \brief Returns a vector of matches.
    #[inline] pub fn get_matches<T,U>(&self, df: &mut NNGraph) -> Vec<MatchedSubgraph<T,U>> {
        
        todo!();
        /*
            std::vector<MatchedSubgraph> matches;
      if (!syntaxIsValid_) {
        return matches;
      }
      // Attempt to match at each node
      for (const auto& node : df.getMutableNodes()) {
        auto match = matchGraph_.isSubgraphMatch(node, matchGraphRootNode_, true);
        if (match.isMatch()) {
          MatchedSubgraph ms;
          ms.subgraph = *match.getMatchedSubgraph();
          // This is a map from the the internal TestMatchGraph to the nodes in the
          // NNGraph
          auto match_graph_map = match.getMatchNodeMap();
          // We iterate through the "varMap_" map (string ->
          // TestMatchGraph_NodeRef) to generate string -> NNGraph_NodeRef
          for (auto p : varMap_) {
            auto iter = match_graph_map->find(p.second);
            if (iter != match_graph_map->end()) {
              ms.matchMap[p.first] = iter->second;
            }
          }
          for (auto p : callMap_) {
            auto iter = match_graph_map->find(p.second);
            if (iter != match_graph_map->end()) {
              ms.matchMap[p.first] = iter->second;
            }
          }
          matches.emplace_back(ms);
        }
      }
      return matches;
        */
    }
}
