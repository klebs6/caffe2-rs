crate::ix!();

use crate::{
    GraphType,
    EdgeRef,
    NodeRef,
    NomGraph,
};

pub struct TestClass { }

pub struct NNEquality { }

impl NNEquality {
    
    #[inline] pub fn equal<T,U>(a: &NodeRef<T,U>, b: &NodeRef<T,U>) -> bool {
        
        todo!();
        /*
            if (
            !nom::repr::nn::is<nom::repr::NeuralNetOperator>(a) ||
            !nom::repr::nn::is<nom::repr::NeuralNetOperator>(b)) {
          return false;
        }
        auto a_ = nom::repr::nn::get<nom::repr::NeuralNetOperator>(a);
        auto b_ = nom::repr::nn::get<nom::repr::NeuralNetOperator>(b);

        bool sameKind = a_->getKind() == b_->getKind();
        if (sameKind && a_->getKind() == nom::repr::NeuralNetOperator::NNKind::GenericOperator) {
          return a_->getName() == b_->getName();
        }
        return sameKind;
        */
    }
}

///----------------------------
const A: u32 = 1103515245;
const C: u32 = 12345;

/**
  | Very simple random number generator
  | used to generate platform independent
  | random test data.
  |
  */
pub struct TestRandom {
    seed: u32,
}

impl TestRandom {
    
    pub fn new(seed: u32) -> Self {
    
        todo!();
        /*
            : seed_(seed)
        */
    }
    
    #[inline] pub fn next_int(&mut self) -> u32 {
        
        todo!();
        /*
            seed_ = A * seed_ + C;
        return seed_;
        */
    }
}

/** 
 | Our test graph looks like this:
 |           +-------+
 |           | entry |
 |           +-------+
 |             |
 |             |
 |             v
 |           +-------+
 |           |   1   |
 |           +-------+
 |             |
 |             |
 |             v
 | +---+     +-------+
 | | 4 | <-- |   2   |
 | +---+     +-------+
 |   |         |
 |   |         |
 |   |         v
 |   |       +-------+
 |   |       |   3   |
 |   |       +-------+
 |   |         |
 |   |         |
 |   |         v
 |   |       +-------+
 |   +-----> |   6   |
 |           +-------+
 |             |
 |             |
 |             v
 | +---+     +-------+
 | | 5 | --> |   7   |
 | +---+     +-------+
 |             |
 |             |
 |             v
 |           +-------+
 |           | exit  |
 |           +-------+
 |
 | Here is the code used to generate the dot file
 | for it:
 |
 |  auto str = nom::converters::convertToDotString(&graph,
 |    [](nom::Graph<std::string>::NodeRef node) {
 |      std::map<std::string, std::string> labelMap;
 |      labelMap["label"] = node->data();
 |      return labelMap;
 |    });
 */
#[inline] pub fn to_string<T>(value: T) -> String {

    todo!();
    /*
        std::ostringstream os;
        os << value;
        return os.str();
    */
}

#[inline] pub fn create_graph() 
-> NomGraph<String> {
    
    todo!();
    /*
        nom::Graph<std::string> graph;
      auto entry = graph.createNode(std::string("entry"));
      auto n1 = graph.createNode(std::string("1"));
      auto n2 = graph.createNode(std::string("2"));
      auto n3 = graph.createNode(std::string("3"));
      auto n4 = graph.createNode(std::string("4"));
      auto n5 = graph.createNode(std::string("5"));
      auto n6 = graph.createNode(std::string("6"));
      auto n7 = graph.createNode(std::string("7"));
      auto exit = graph.createNode(std::string("exit"));
      graph.createEdge(entry, n1);
      graph.createEdge(n1, n2);
      graph.createEdge(n2, n3);
      graph.createEdge(n2, n4);
      graph.createEdge(n3, n6);
      graph.createEdge(n4, n6);
      graph.createEdge(n6, n7);
      graph.createEdge(n5, n7);
      graph.createEdge(n7, exit);
      return graph;
    */
}


#[inline] pub fn create_graph_with_cycle() 
-> NomGraph<String> {

    todo!();
    /*
        nom::Graph<std::string> graph;
      auto entry = graph.createNode(std::string("entry"));
      auto n1 = graph.createNode(std::string("1"));
      auto n2 = graph.createNode(std::string("2"));
      auto n3 = graph.createNode(std::string("3"));
      auto n4 = graph.createNode(std::string("4"));
      auto n5 = graph.createNode(std::string("5"));
      auto n6 = graph.createNode(std::string("6"));
      auto n7 = graph.createNode(std::string("7"));
      auto exit = graph.createNode(std::string("exit"));
      graph.createEdge(entry, n1);
      graph.createEdge(n1, n2);
      graph.createEdge(n2, n3);
      graph.createEdge(n2, n4);
      graph.createEdge(n3, n6);
      graph.createEdge(n6, n3); // Cycle
      graph.createEdge(n4, n6);
      graph.createEdge(n6, n7);
      graph.createEdge(n5, n7);
      graph.createEdge(n7, exit);
      return graph;
    */
}

#[inline] pub fn bb_printer<T,U>(node: NodeRef<T,U>) 
-> HashMap<String,String> 
{
    todo!();
    /*
        std::map<std::string, std::string> labelMap;
      auto& bb = node->data();
      labelMap["label"] = to_string((unsigned long long)node) + "\\n";
      for (const auto& instr : bb.getInstructions()) {
        assert(isa<nom::repr::NeuralNetOperator>(instr->data()) &&
               "Invalid instruction.");
        auto *op = dyn_cast<nom::repr::NeuralNetOperator>(instr->data().get());
        bool hasOutput = false;
        for (const auto &outEdge : instr->getOutEdges()) {
          auto *output =
              dyn_cast<nom::repr::NeuralNetData>(outEdge->head()->data().get());
          labelMap["label"] += " " + output->getName();
          hasOutput = true;
        }
        if (hasOutput) {
          labelMap["label"] += " = ";
        }
        labelMap["label"] += op->getName();
        for (const auto &inEdge : instr->getInEdges()) {
          auto *arg =
              dyn_cast<nom::repr::NeuralNetData>(inEdge->tail()->data().get());
          labelMap["label"] += " " + arg->getName();
        }
        labelMap["label"] += "\\l";
      }
      labelMap["shape"] = "box";
      return labelMap;
    */
}

#[inline] pub fn cfg_edge_printer<T,U>(edge: EdgeRef<T,U>) 
-> HashMap<String,String> 
{
    todo!();
    /*
        std::map<std::string, std::string> labelMap;
      if (edge->data() == -1) {
        labelMap["label"] = "F";
      } else if (edge->data() == 1) {
        labelMap["label"] = "T";
      }
      return labelMap;
    */
}

#[inline] pub fn nn_printer<T,U>(node: NodeRef<T,U>) 
-> HashMap<String,String> 
{
    
    todo!();
    /*
        std::map<std::string, std::string> labelMap;
      assert(node->data() && "Node doesn't have data, can't render it");
      if (isa<nom::repr::NeuralNetOperator>(node->data())) {
        auto *op = dyn_cast<nom::repr::NeuralNetOperator>(node->data().get());
        labelMap["label"] =
            op->getName() + " (" + to_string((unsigned long long)node) + ")";
        labelMap["shape"] = "box";
      } else if (isa<nom::repr::Data>(node->data())) {
        auto tensor = dyn_cast<nom::repr::NeuralNetData>(node->data().get());
        labelMap["label"] = tensor->getName();
        labelMap["label"] += "_" + to_string(tensor->getVersion()) + " " + to_string((unsigned long long)node);
      }
      return labelMap;
    */
}

pub fn create_test_node(g: &mut NomGraph<TestClass>) 
    -> <NomGraph::<TestClass> as GraphType>::NodeRef 
{
    todo!();
    /*
       return g.createNode(TestClass());
       */
}

pub fn test_node_printer(unused: <NomGraph<TestClass> as GraphType>::NodeRef) 
    -> HashMap<String, String> 
{
    todo!();
    /*
       std::map<std::string, std::string> labelMap;
       labelMap["label"] = "Node";
       return labelMap;
       */
}
