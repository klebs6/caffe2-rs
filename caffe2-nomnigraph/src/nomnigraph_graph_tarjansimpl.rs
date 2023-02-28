crate::ix!();

use crate::{
    GraphType,
    Subgraph,
    NomGraph,
    EdgeRef,
    NodeRef,
    Graph,
};

pub type EdgeWrapper<T,U>     = <GraphWrapper::<T, U> as HasEdgeWrapper>::EdgeWrapper;
pub type WrappedGraph<T,U>    = NomGraph::<NodeWrapper<T,U>, EdgeWrapper<T,U>>;
pub type WrappedSubgraph<T,U> = Subgraph::<NodeWrapper<T,U>, EdgeWrapper<T,U>>;

impl<T,U> GraphType for WrappedGraph<T,U> {
    type NodeRef = NodeRef<T,U>;
    type EdgeRef = EdgeRef<T,U>;
    type SubgraphType = WrappedSubgraph<T,U>;
}

pub struct NodeWrapper<T,U> {
    node:     NodeRef<T,U>,
    index:    i32, // default = -1
    low_link: i32, // default = -1
    on_stack: bool, // default = false
}

impl<T,U> NodeWrapper<T,U> {
    
    pub fn new(n: NodeRef<T,U>) -> Self {
    
        todo!();
        /*
            : node(n)
        */
    }
}

pub struct GraphWrapper<T, U> {
    phantomA: PhantomData<T>,
    phantomB: PhantomData<U>,
}

impl<T,U> HasEdgeWrapper for GraphWrapper<T,U> {
    type EdgeWrapper = i32;//TODO where is the proper type for EdgeWrapper?
}

pub trait HasEdgeWrapper {
    type EdgeWrapper;
}

/**
  | -----------
  | @brief
  | 
  | Tarjans algorithm implementation.
  | 
  | See details on how the algorithm works
  | here: https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
  | 
  | The algorithm works by annotating nodes,
  | but we want to be able to handle generic
  | graphs. Thus, we wrap the input graph
  | with nodes that contain data composed
  | of references to the original graph
  | (for later recovery) and the data required
  | for the algorithm (see NodeWrapper).
  | 
  | We then run the algorithm and return
  | a reverse-topologically sorted vector
  | of strongly connect components in the
  | form of Subgraphs on the Graph.
  | 
  | -----------
  | @note
  | 
  | Head/Tail is used in reverse in Tarjan's
  | early papers.
  | 
  | \bug Edges not included in returned
  | subgraphs.
  |
  */
pub struct Tarjans<T, U> {

    index:                i32, // default = 0
    stack:                Vec<<WrappedGraph<T,U> as GraphType>::NodeRef>,
    input_graph:          *mut NomGraph<T,U>,
    wrapped_input_graph:  WrappedGraph<T,U>,
    wrapped_sccs:         Vec<WrappedSubgraph<T,U>>,
}

impl<T, U> Tarjans<T, U> {
    
    /**
      | \brief Constructor wraps the input graph
      |        with an annotated graph set up with
      |        the datastructures needed for the
      |        algorithm.
      |
      | \p g The graph Tarjan's will be run on.
      */
    pub fn new(g: *mut NomGraph<T,U>) -> Self {
    
        todo!();
        /*
            : InputGraph(g) 

        // Wrap Graph with node labels
        std::unordered_map<
            typename Graph<T, U>::NodeRef,
            typename WrappedGraph::NodeRef>
            n_to_wrappedNode;

        for (const auto& n : InputGraph->getMutableNodes()) {
          NodeWrapper wrappedNode(n);
          n_to_wrappedNode[n] =
              WrappedInputGraph.createNode(std::move(wrappedNode));
        }

        for (const auto& e : InputGraph->getMutableEdges()) {
          EdgeWrapper wrappedEdge = {e};
          WrappedInputGraph.createEdge(
              n_to_wrappedNode[e->tail()],
              n_to_wrappedNode[e->head()],
              std::move(wrappedEdge));
        }
        */
    }

    /**
      | \brief Helper function for finding
      | strongly connected components.
      |
      | \p n A reference to a node within the
      | wrapped graph.
      */
    #[inline] pub fn connect(&mut self, n: <WrappedGraph<T,U> as GraphType>::NodeRef)  {
        
        todo!();
        /*
            n->mutableData()->Index = Index;
        n->mutableData()->LowLink = Index;
        Index++;

        Stack.emplace_back(n);
        n->mutableData()->OnStack = true;

        for (const auto& outEdge : n->getOutEdges()) {
          typename WrappedGraph::NodeRef newNode = outEdge->head();
          // Check if we've considered this node before.
          if (newNode->data().Index == -1) {
            connect(newNode);
            n->mutableData()->LowLink =
                std::min(n->data().LowLink, newNode->data().LowLink);
            // Check if newNode is in the SCC.
          } else if (newNode->data().OnStack) {
            n->mutableData()->LowLink =
                std::min(n->data().LowLink, newNode->data().Index);
          }
        }

        // If our node is a root node, pop it from the stack (we've found an SCC)
        if (n->data().LowLink == n->data().Index) {
          WrappedSubgraph wrappedSCC;

          typename WrappedGraph::NodeRef w;
          do {
            w = Stack.back();
            w->mutableData()->OnStack = false;
            Stack.pop_back();
            wrappedSCC.addNode(w);
          } while (w != n);

          // Add all the edges into the SCC.
          // TODO include edges in the SCC in a smarter way.
          const auto& sccNodes = wrappedSCC.getNodes();
          for (const auto& sccNode : sccNodes) {
            for (const auto& outEdge : sccNode->getOutEdges()) {
              if (std::find(sccNodes.begin(), sccNodes.end(), outEdge->head()) !=
                  sccNodes.end()) {
                wrappedSCC.addEdge(outEdge);
              }
            }
          }
          WrappedSCCs.emplace_back(wrappedSCC);
        }
        */
    }

    /**
      | \brief Helper function for recovering
      | a valid subgraph output.
      |
      | \p wrappedS A wrapped subgraph.
      |
      | \return A subgraph of the original input
      | graph.
      |
      */
    #[inline] pub fn unwrap_subgraph(&mut self, 
        wrapped_subgraph: &WrappedSubgraph<T,U>) -> Subgraph<T,U> 
    {
        todo!();
        /*
            Subgraph<T, U> s;
        for (auto wrappedNode : wrappedSubgraph.getNodes()) {
          s.addNode(wrappedNode->data().node);
        }
        for (auto wrappedEdge : wrappedSubgraph.getEdges()) {
          s.addEdge(wrappedEdge->data().edge);
        }
        return s;
        */
    }
    
    #[inline] pub fn run(&mut self) -> Vec<Subgraph<T,U>> {
        
        todo!();
        /*
            for (auto n : WrappedInputGraph.getMutableNodes()) {
          if (n->data().Index == -1) {
            connect(n);
          }
        }

        std::vector<Subgraph<T, U>> sccs;
        for (auto wrappedSCC : WrappedSCCs) {
          sccs.emplace_back(unwrapSubgraph(wrappedSCC));
        }

        return sccs;
        */
    }
}

/**
  | -----------
  | @brief
  | 
  | A function wrapper to infer the graph
  | template parameters.
  |
  */
#[inline] pub fn tarjans<T, U>(g: *mut NomGraph<T,U>) -> Vec<Subgraph<T,U>> {

    todo!();
    /*
        Tarjans<T, U> t(g);
      return t.run();
    */
}
