crate::ix!();

/**
  | Graph representation of an operator.
  | 
  | Default constructor for resize
  |
  */
#[derive(Default)]
pub struct Node {

    /**
      | The OperatorDef which this node represents.
      |
      */
    op: OperatorDef,

    /**
      | Keeps track of if an operator has been
      | deleted through a transformation.
      |
      */
    active: bool, //true

    /**
      | Stores a pair (idx, blob_list),
      |
      |  idx = index of the child
      |
      |  blob_list = a list of strings, containing
      |  the blobs that connect the nodes
      */
    parents:  HashMap<i32,Vec<String>>,
    children: HashMap<i32,Vec<String>>,
}

impl Node {
    
    /// Alternate constructor
    pub fn new(
        op:       &OperatorDef,
        active:   bool,
        parents:  HashMap<i32,Vec<String>>,
        children: HashMap<i32,Vec<String>>) -> Self 
    {
        todo!();
        /*
            : op(op), active(active), parents(parents), children(children)
        */
    }
}

/**
  | Graph representation of a Netdef.
  |
  */
pub struct Graph {

    /**
      | Stores the netdef representation.
      | 
      | Is updated upon calls to GetNetDef.
      |
      */
    netdef: NetDef,

    /**
      | Stores which blobs the graph reads from,
      | and writes to.
      |
      */
    external_input:  HashSet<String>,
    external_output: HashSet<String>,

    /**
      | Keeps track of all the Operators currently
      | within graph, even if inactive.
      |
      */
    nodes: Vec<Node>,
}

impl Graph {

    #[inline] pub fn size(&self) -> usize {
        
        todo!();
        /*
            return nodes_.size();
        */
    }
    
    #[inline] pub fn push_node(&mut self, new_node: &Node)  {
        
        todo!();
        /*
            return nodes_.push_back(new_node);
        */
    }
    
    #[inline] pub fn resize_nodes(&mut self, new_size: usize)  {
        
        todo!();
        /*
            nodes_.resize(new_size);
        */
    }

    /**
      | Index safe, less verbose way to access
      | nodes
      |
      */
    #[inline] pub fn node<'a>(&'a self, idx: usize) -> &'a Node {
        
        todo!();
        /*
            return nodes_.at(idx);
        */
    }
    
    #[inline] pub fn node_mut<'a>(&'a mut self, idx: usize) -> &'a mut Node {
        
        todo!();
        /*
            return nodes_.at(idx);
        */
    }
    
    #[inline] pub fn is_node_active(&mut self, idx: usize) -> bool {
        
        todo!();
        /*
            return node(idx).active;
        */
    }
    
    #[inline] pub fn external_input(&self) -> &HashSet<String> {
        
        todo!();
        /*
            return external_input_;
        */
    }
    
    #[inline] pub fn external_output(&self) -> &HashSet<String> {
        
        todo!();
        /*
            return external_output_;
        */
    }
    
    /**
      | Graph generation.
      | 
      | Given a netdef, returns a Graph.
      | 
      | Each node represents an operator.
      | 
      | An edge exists between two nodes if the
      | parent op writes to a blob, which is the
      | input of the child blob, with no other
      | op writing to the blob in between the
      | execution order.
      | 
      | Time Complexity: O(E), where E is the
      | number of blobs
      |
      */
    pub fn new(net: &NetDef) -> Self {
        todo!();
        /*
            : netdef_(net) 

      nodes_.clear();
      nodes_.resize(net.op_size());

      // Copy over operators
      for (int x = 0; x < net.op_size(); x++) {
        node(x).op = net.op(x);
      }

      // For any blob, which operator was the last to write to it?
      // In python, this is known as "versions".
      std::unordered_map<string, int> edge_parent;

      for (int i = 0; i < (int)nodes_.size(); i++) {
        for (const string& blob : node(i).op.input()) {
          auto it = edge_parent.find(blob);
          if (it != edge_parent.end()) {
            int j = it->second;
            node(i).parents[j].push_back(blob);
            node(j).children[i].push_back(blob);
          } else {
            external_input_.insert(blob);
          }
        }
        for (const string& blob : node(i).op.output()) {
          edge_parent[blob] = i;
        }
      }

      // Traverse opposite direction to find external outputs

      // For any blob, which operator was the last to read to from it?
      std::unordered_map<string, int> edge_child;

      for (int i = (int)nodes_.size() - 1; i >= 0; i--) {
        for (const string& blob : node(i).op.output()) {
          auto it = edge_child.find(blob);
          if (it == edge_child.end()) {
            external_output_.insert(blob);
          }
        }
        for (const string& blob : node(i).op.input()) {
          edge_child[blob] = i;
        }
      }
        */
    }
    
    /**
      | Given a subgraph, gets all of the parents
      | of the subgraph, as well as their associated
      | blob names. Sorted by blob names.
      | 
      | <string, int> := (name of blob writing
      | into subgraph, index of node that writes
      | into subgraph using that blob)
      |
      */
    #[inline] pub fn get_subgraph_input(&mut self, match_: &Vec<i32>) -> Vec<(String,i32)> {
        
        todo!();
        /*
            return GetSubgraphPerimeterHelper(true, match);
        */
    }
    
    /**
      | Given a subgraph, gets all of the children
      | of the subgraph, as well as their associated
      | blob names. Sorted by blob names.
      | 
      | <string, int> := (name of blob reading
      | from subgraph, index of node that reads
      | from subgraph using that blob)
      |
      */
    #[inline] pub fn get_subgraph_output(&mut self, 
        match_: &Vec<i32>) -> Vec<(String,i32)> 
    {
        todo!();
        /*
            return GetSubgraphPerimeterHelper(false, match);
        */
    }

    /**
      | This helper function will either get:
      | 
      | 1) a list for the blobs that write INTO
      | a subgraph
      | 
      | 2) a list of for the blobs that are written
      | FROM a subgraph.
      | 
      | The "from_children" flag determines
      | if it is case 1 (true) or case 2 (false).
      |
      */
    #[inline] pub fn get_subgraph_perimeter_helper(
        &mut self, 
        from_children: bool,
        match_: &Vec<i32>) -> Vec<(String,i32)> 
    {
        todo!();
        /*
            std::vector<std::pair<string, int>> edge_list;
      std::unordered_set<int> match_set(match.begin(), match.end());
      for (int x = 0; x < (int)nodes_.size(); x++) {
        if (!is_node_active(x)) {
          continue;
        }
        if (!match_set.count(x)) { // x is not in subgraph
          const auto& list = from_children ? node(x).children : node(x).parents;
          for (const auto& edge : list) {
            int parent = edge.first;
            const auto& blobs = edge.second;
            if (match_set.count(parent)) { // but has a parent that is in subgraph
              for (const string& blob : blobs) {
                edge_list.push_back({blob, x});
              }
            }
          }
        }
      }
      // return the list in sorted order, to allow binary searching
      std::sort(edge_list.begin(), edge_list.end());
      return edge_list;
        */
    }
    
    /**
      | Generates a NetDef Representation
      | for the current graph.
      | 
      | Nodes are visited in topological order,
      | which is proper Opdef ordering.
      | 
      | TODO(benz):
      | 
      | There exists conflicts with repeated
      | blob names, where topological sorting
      | is not sufficient for correct netdef
      | representation, unless blobs are renamed.
      | 
      | For example, if after a transformation,
      | We have operator ancestry:
      | 
      | A --> B --> C, and also A --> D --> E, where
      | B -> C and D -> E uses the same blob name,
      | then A, B, D, E, C is a correct topological
      | ordering, but D will write to the blob
      | that C reads from, instead of B.
      | 
      | Currently believe that there will always
      | be ambiguity unless blobs are renamed.
      | 
      | This is solved by performing SSA on all
      | transformed blob names.
      |
      */
    #[inline] pub fn get_net_def(&mut self) -> NetDef {
        
        todo!();
        /*
            std::vector<bool> visited(nodes_.size(), false);

      // Copy over all the properties of the netdef we're based on
      NetDef netdef = netdef_;

      // But we're going to put in our own operators.
      netdef.clear_op();

      // Keeps track of the number of parents yet to be processed.
      std::vector<int> unchecked_parent_count;

      // We will perform a topological traversal on the nodes, but we will prefer
      // nodes that come earlier in the execution order.

      // This is a min-heap, which stores its elements in ascending order.
      // This stores the nodes in the order we process them to be in.
      // This guarantees the lowest lexicographical topological ordering.

      // This also means the original nodes will be kept in their execution order.
      std::priority_queue<int, std::vector<int>, std::greater<int>> q;

      // In our graph, G, the nodes don't have a strict ordering. But in the netdef,
      // they must (since nets are operators executed in some order).
      // How do we make sure that the order of operators in our generated netdef
      // is valid?
      // 1) The ordering of the netdef must be topologically sorted, respect to G.
      //    If A -> B is an edge in the graph G, then A must come before B in the
      //    netdef's ordering.
      // 2) No blob conflicts: If A -> B is an edge in the graph G, and A writes to
      //    blob X and B reads from blob X, then there cannot be an op that writes
      //    to blob X between A and B in the ordering.
      //
      // Perform a Topological Sort, to find an order for the Operators to be in.
      // We will keep track of the number of parents each node has.
      // We begin with an empty queue, and push in all nodes that do not have any
      // parents. Then, we keep track of all unprocessed parents for each node.
      // When a node has no more unprocessed parents, we can push it into the queue
      // to be processed. This guarantees condition 1 is satisfied.

      // TODO(benz): Currently, condition 2 is not guaranteed to be satisified.
      // However, giving each blob unique names via SSA will satisfy this condition.
      // Then, the resulting graph can be optimized with memonger.

      for (int i = 0; i < (int)nodes_.size(); i++) {
        unchecked_parent_count.push_back(node(i).parents.size());
        if (node(i).parents.size() == 0 && is_node_active(i)) {
          q.push(i);
          visited[i] = true;
        }
      }

      while (!q.empty()) {
        int idx = q.top();
        q.pop();
        if (!is_node_active(idx)) {
          continue;
        }
        // Creates a new OperatorDef in NetDef
        auto& op = *(netdef.add_op());
        // Sets it equal to the OperatorDef at node(idx)
        op = node(idx).op;
        for (const auto& edge : node(idx).children) {
          int child = edge.first;
          if (!visited[child] && is_node_active(child)) {
            unchecked_parent_count[child]--;
            if (unchecked_parent_count[child] == 0) {
              q.push(child);
              visited[child] = true;
            }
          }
        }
      }
      return netdef;
        */
    }
    
    /**
      | Deactivate a subgraph, and get rid of
      | all edges into this subgraph.
      |
      */
    #[inline] pub fn deactivate_subgraph(&mut self, subgraph: Vec<i32>)  {
        
        todo!();
        /*
            for (int idx : subgraph) {
        // remove all edges connected to inactive node
        for (const auto& edge : node(idx).parents) {
          int parent = edge.first;
          node(parent).children.erase(idx);
        }
        for (const auto& edge : node(idx).children) {
          int child = edge.first;
          node(child).parents.erase(idx);
        }
        // actually mark flags as false
        node(idx).active = false;
      }
        */
    }
}

/**
  | Adds an operator def to a netdef.
  |
  | Returns the ptr, if you want to add anything
  | extra (such as device_option)
  */
#[inline] pub fn add_op(
    netdef_ptr: *mut NetDef,
    op_type:    String,
    inputs:     Vec<String>,
    outputs:    Vec<String>) -> *mut OperatorDef 
{

    todo!();
    /*
        CHECK(netdef_ptr);
      auto& netdef = *netdef_ptr;
      auto op_ptr = netdef.add_op();
      auto& op = *op_ptr;
      op.set_type(op_type);
      for (const string& inp : inputs) {
        op.add_input(inp);
      }
      for (const string& outp : outputs) {
        op.add_output(outp);
      }
      return op_ptr;
    */
}

/**
  | This allows for the use of * and | to match
  | operator types, engines, or any other
  | property that is represented by strings.
  | 
  | For example, if we wanted to match an
  | operator to Conv or FC, we can give: "Conv|FC"
  | as the type() of that op.
  |
  */
#[inline] pub fn match_strings(
    p: String,
    s: String) -> bool 
{
    todo!();
    /*
        if (p == "*") { // star accepts anything
        return true;
      }
      // TODO(benz): memoize this. (high constant factor boost in performance)
      vector<string> choices = split('|', p);
      for (const string& candidate : choices) {
        if (candidate == s) {
          return true;
        }
      }
      return false;
    */
}

/**
  | This ensures that each named arg that
  | exists in the pattern exists in g_op,
  | is equal in value.
  |
  */
#[inline] pub fn match_arguments(
    p_op: &OperatorDef,
    g_op: &OperatorDef) -> bool 
{
    todo!();
    /*
        for (const auto& p_arg : p_op.arg()) {
        if (!p_arg.has_name()) {
          continue;
        }
        bool found = false;
        for (const auto& g_arg : g_op.arg()) {
          if (p_arg.name() == g_arg.name()) {
            found = true;
            if (p_arg.has_f()) {
              if (!g_arg.has_f() || p_arg.f() != g_arg.f()) {
                return false;
              }
            }
            if (p_arg.has_i()) {
              if (!g_arg.has_i() || p_arg.i() != g_arg.i()) {
                return false;
              }
            }
            if (p_arg.has_s()) {
              if (!g_arg.has_s() || !MatchStrings(p_arg.s(), g_arg.s())) {
                return false;
              }
            }
            if (p_arg.floats_size() != g_arg.floats_size()) {
              return false;
            }
            for (int i = 0; i < p_arg.floats_size(); i++) {
              if (p_arg.floats(i) != g_arg.floats(i)) {
                return false;
              }
            }
            if (p_arg.ints_size() != g_arg.ints_size()) {
              return false;
            }
            for (int i = 0; i < p_arg.ints_size(); i++) {
              if (p_arg.ints(i) != g_arg.ints(i)) {
                return false;
              }
            }
            if (p_arg.strings_size() != g_arg.strings_size()) {
              return false;
            }
            for (int i = 0; i < p_arg.strings_size(); i++) {
              if (!MatchStrings(p_arg.strings(i), g_arg.strings(i))) {
                return false;
              }
            }
          }
        }
        if (!found) {
          return false;
        }
      }
      return true;
    */
}

