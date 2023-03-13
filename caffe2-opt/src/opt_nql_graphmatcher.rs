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

/**
  | Helper function for convertToNQLString
  | function.
  |
  | Given a node and a renameMap return the unique
  | name for this node.
  */
#[inline] pub fn get_name_for_blob(node: NNGraph_NodeRef, rename_map: &HashMap<NNGraph_NodeRef,String>) -> String {
    
    todo!();
    /*
        if (renameMap.count(node)) {
        return renameMap.at(node);
      }
      return getNodeName(node);
    */
}

/**
  | Helper function for convertToNQLString
  | function.
  |
  | Given a node and a renameMap return a string
  | representing the node, which looks something
  | like:
  |
  |   %a = Op(%b, %c, %d)
  */
#[inline] pub fn get_nqlstring_for_blob(
    node: NNGraph_NodeRef, 
    rename_map: &HashMap<NNGraph_NodeRef,String>) -> String 
{
    
    todo!();
    /*
        if (!nn::is<Data>(node) || !nn::hasProducer(node)) {
        return "";
      }
      NNGraph_NodeRef defOp = nn::getProducer(node);

      std::string result =
          getNameForBlob(node, renameMap) + " = " + getNodeName(defOp) + "(";
      int i = 0;
      for (auto inputTensor : nn::getInputs(defOp)) {
        if (i) {
          result += ", ";
        }
        result += getNameForBlob(inputTensor, renameMap);
        i++;
      }
      result += ")";
      return result;
    */
}

/**
  | Helper function for convertToNQLString
  | function.
  |
  | It takes a list of nodes and returns a map
  | node->unique_name. The new names are based on
  | the existing ones, but are also unique.
  */
#[inline] pub fn compute_dedup_rename_map(nodes: &Vec<NNGraph_NodeRef>) -> HashMap<NNGraph_NodeRef,String> {
    
    todo!();
    /*
        std::unordered_map<NNGraph_NodeRef, std::string> renameMap;
      std::unordered_set<std::string> takenNames;
      takenNames.clear();
      for (auto node : nodes) {
        std::string name = getNodeName(node);
        if (!isa<Data>(node->data())) {
          continue;
        }
        std::string newName = name;
        int dedupCounter = 0;
        while (takenNames.count(newName)) {
          newName = name + "_" + caffe2::to_string(dedupCounter);
          dedupCounter++;
        }
        renameMap[node] = newName;
        takenNames.insert(newName);
      }
      return renameMap;
    */
}

/**
  | \brief Return a short string name for the
  | given \param node.
  |
  | The function works with both tensors and
  | operators.
  */
#[inline] pub fn get_node_name(node: NNGraph_NodeRef) -> String {
    
    todo!();
    /*
        if (!node) {
        return "";
      }
      if (nn::is<NeuralNetOperator>(node)) {
        if (auto* op = nn::get<NeuralNetOperator>(node)) {
          return op->getName();
        }
      }
      if (nn::is<NeuralNetData>(node)) {
        if (auto tensor = nn::get<NeuralNetData>(node)) {
          return "%" + tensor->getName();
        }
      }
      return "";
    */
}

/**
  | \brief Return a string representing the given
  | graph \param g.
  |
  | The returned string is a valid NQL query.
  */
#[inline] pub fn convert_to_nqlstring(g: &mut NNGraph) -> String {
    
    todo!();
    /*
        // Order nodes in a topological order.
      // TODO: Currently tarjans mutates the graph, and that's the only reason we
      // are not using const reference for `g`. We need to fix tarjans so that it
      // doesn't mutate the graph and use const reference in this function too.
      auto topoMatch = nom::algorithm::tarjans(&g);
      std::vector<NNGraph_NodeRef> nodes;
      int sccNum = 0;
      for (auto scc : topoMatch) {
        sccNum++;
        for (auto node : scc.getNodes()) {
          nodes.emplace_back(node);
        }
      }
      std::reverse(nodes.begin(), nodes.end());

      // Different nodes might have the same name. We want to change that so that
      // they are distinguishable by the name. NQL assumes that names are unique.
      std::unordered_map<NNGraph_NodeRef, std::string> renameMap =
          computeDedupRenameMap(nodes);

      // Going from top to bottom (nodes are in topological order), print all
      // nodes.
      std::string result = "def nn {\n";
      for (auto node : nodes) {
        std::string r = getNQLStringForBlob(node, renameMap);
        if (!r.empty()) {
          result += "  " + r + "\n";
        }
      }
      result += "}\n";
      return result;
    */
}
