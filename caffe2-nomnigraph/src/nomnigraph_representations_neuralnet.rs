crate::ix!();

pub enum AnnotationKind { 
    Generic, 
    Caffe2 
}

/**
  | \brief Annotations allow for generic
  | manipulation of neural network operations.
  | The base class contains a saved void* pointer
  | for external use.  Derived classes add richer
  | semantics to the annotation and it is
  | encouraged to use them.
  */
pub struct Annotation {
    kind:  AnnotationKind,
}

impl Annotation {
    
    pub fn new(kind: AnnotationKind) -> Self {
    
        todo!();
        /*
            : kind_(kind)
        */
    }
    
    #[inline] pub fn get_kind(&self) -> AnnotationKind {
        
        todo!();
        /*
            return kind_;
        */
    }
}

impl Default for Annotation {
    
    fn default() -> Self {
        todo!();
        /*
            : kind_(AnnotationKind::Generic
        */
    }
}

///--------------------------
/// Discriminator for LLVM-style RTTI (isa<>)
pub enum NNKind {
    Undefined,
    GenericOperator,
    NNPhi,
    While,
    Relu, 
    Conv, 
    ConvRelu, 
    ConvTranspose, 
    AveragePool, 
    AveragePoolRelu, 
    MaxPool,
    MaxPoolRelu, 
    Sum, 
    SumRelu, 
    Send, 
    Receive, 
    BatchNormalization, 
    Clip, 
    FC,
    GivenTensorFill, 
    Concat, 
    Softmax, 
    ChannelShuffle, 
    Add, 
    Reshape, 
    Flatten,
    CopyToOpenCL, 
    CopyFromOpenCL, 
    NCHW2NHWC, 
    NHWC2NCHW, 
    Declare, 
    Export
}

/// An optional tensor-type specifier.
pub enum NNLayout { 
    Undefined, 
    NCHW, 
    NHWC
}

pub struct NeuralNetOperator {
    base: Instruction,

    kind:              NNKind,

    /// Mutable attribute, much like a type cast
    layout:            NNLayout,

    extra_annotation:  Box<Annotation>,
}

impl Default for NeuralNetOperator {
    
    fn default() -> Self {
        todo!();
        /*
            : Instruction(), kind_(NNKind::Undefined), layout_(NNLayout::Undefined
        */
    }
}

impl From<NNKind> for NeuralNetOperator {

    fn from(k: NNKind) -> Self {
    
        todo!();
        /*
            : Instruction(), kind_(K), layout_(NNLayout::Undefined)
        */
    }
}

impl Named for NeuralNetOperator {
    
    #[inline] fn get_name(&self) -> String {
        
        todo!();
        /*
            switch (getKind()) {
    #include "nomnigraph/Generated/OpNames.h"
        case NNKind::While:
          return "While";
        case NNKind::NNPhi:
          return "Phi";
        case NNKind::GenericOperator:
          return dyn_cast<GenericOperator>(this)->getName();
        default:
          return "Unknown";
      }
        */
    }
}

impl NeuralNetOperator {
    
    pub fn new_from_kind_opcode_layout(k: NNKind, i: Opcode, l: NNLayout) -> Self {

        todo!();
        /*
            : Instruction(I), kind_(K), layout_(L)
        */
    }
    
    pub fn new_from_kind_and_opcode(k: NNKind, i: Opcode) -> Self {
    
        todo!();
        /*
            : Instruction(I), kind_(K), layout_(NNLayout::Undefined)
        */
    }
    
    pub fn new_from_kind_and_layout(k: NNKind, l: NNLayout) -> Self {
    
        todo!();
        /*
            : Instruction(), kind_(K), layout_(L)
        */
    }
    
    #[inline] pub fn get_kind(&self) -> NNKind {
        
        todo!();
        /*
            return kind_;
        */
    }
    
    #[inline] pub fn set_layout(&mut self, l: NNLayout)  {
        
        todo!();
        /*
            layout_ = L;
        */
    }
    
    #[inline] pub fn get_layout(&self) -> NNLayout {
        
        todo!();
        /*
            return layout_;
        */
    }
    
    #[inline] pub fn set_annotation(
        &mut self, 
        extra_annotation: Box<Annotation>)
    {
        
        todo!();
        /*
            extraAnnotation_ = std::move(extraAnnotation);
        */
    }
    
    #[inline] pub fn get_annotation(&self) -> *const Annotation {
        
        todo!();
        /*
            return extraAnnotation_.get();
        */
    }
    
    #[inline] pub fn get_mutable_annotation(&mut self) -> *mut Annotation {
        
        todo!();
        /*
            return extraAnnotation_.get();
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Validate the inputs and outputs to this
      | operator.
      | 
      | \p inputs A vector of references to
      | 
      | NeuralNetData types that represent
      | the data being fed into the operator.
      | 
      | \p outputs A vector of references to
      | 
      | NeuralNetData types that represent
      | the data being outputted by the operator.
      | 
      | 
      | -----------
      | @return
      | 
      | true if the inputs and outputs are compatible
      | with the operator.
      |
      */
    #[inline] pub fn check_inputs_and_outputs(
        &mut self, 
        inputs:  Vec<*const NeuralNetData>, 
        outputs: Vec<*const NeuralNetData>) -> bool 
    {

        todo!();
        /*
            return true;
        */
    }
}

///-----------------------
/// Discriminator for LLVM-style RTTI (isa<>)
pub enum NNDataKind { 
    Generic, 
    Tensor 
}

pub struct NeuralNetData {
    base: Data,
    kind: NNDataKind,
}

impl Default for NeuralNetData {
    
    fn default() -> Self {
        todo!();
        /*
            : kind_(NNDataKind::Generic
        */
    }
}

pub trait NeuralNetDataTrait {
    fn clone(&mut self) -> *mut NeuralNetData;
}

impl NeuralNetData {

    pub fn new(kind: NNDataKind) -> Self {
    
        todo!();
        /*
            : kind_(kind)
        */
    }
    
    #[inline] pub fn get_kind(&self) -> NNDataKind {
        
        todo!();
        /*
            return kind_;
        */
    }
}

impl Named for NeuralNetData {
    
    #[inline] fn get_name(&self) -> String {
        
        todo!();
        /*
            switch (getKind()) {
        case NNDataKind::Tensor: {
          return dyn_cast<Tensor>(this)->getName();
        }
        default:
          return "";
      }
        */
    }
}


///-------------------------
pub enum DataType { 
    Generic, 
    Float, 
    Half, 
    Int8 
}

pub enum Layout { 
    Generic, 
    NCHW, 
    NHWC 
}

pub struct NomTensor {
    base: NeuralNetData,
    name:  String,
    type_: DataType,
}

impl NomTensor {

    pub fn new(name: String) -> Self {
    
        todo!();
        /*
            : NeuralNetData(NNDataKind::Tensor),
            name_(name),
            type_(DataType::Generic)
        */
    }
    
    #[inline] pub fn classof(d: *const NeuralNetData) -> bool {
        
        todo!();
        /*
            return D->getKind() == NNDataKind::Tensor;
        */
    }
    
    #[inline] pub fn clone(&mut self) -> *mut NeuralNetData {
        
        todo!();
        /*
            return new NomTensor(name_);
        */
    }
    
    #[inline] pub fn set_type(&mut self, ty: DataType)  {
        
        todo!();
        /*
            type_ = type;
        */
    }
    
    #[inline] pub fn get_type(&self) -> DataType {
        
        todo!();
        /*
            return type_;
        */
    }
    
    #[inline] pub fn set_name(&mut self, name: &String)  {
        
        todo!();
        /*
            name_ = name;
        */
    }
}

impl Named for NomTensor {

    #[inline] fn get_name(&self) -> String {
        
        todo!();
        /*
            return name_;
        */
    }
}

#[macro_export] macro_rules! nomnigraph_define_nn_rtti {
    ($op:ident) => {
        todo!();
        /*
        
          static bool classof(const NeuralNetOperator* N) {        
            return N->getKind() == NNKind::op;                                
          }                                                                   
          static bool classof(const Value* N) {                    
            if (isa<NeuralNetOperator>(N)) {                                  
              return dyn_cast<NeuralNetOperator>(N)->getKind() == NNKind::op; 
            }                                                                 
            return false;                                                     
          }
        */
    }
}

///--------------------------
pub struct While {
    //NOMNIGRAPH_DEFINE_NN_RTTI(While);
    base: NeuralNetOperator,
}

impl Default for While {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::While, Opcode::Branch
        */
    }
}

///--------------------------
pub struct NNPhi {
    //NOMNIGRAPH_DEFINE_NN_RTTI(NNPhi);
    base: NeuralNetOperator,
}
impl Default for NNPhi {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::NNPhi, Opcode::Phi
        */
    }
}

///--------------------------
pub struct GenericOperator {
    //NOMNIGRAPH_DEFINE_NN_RTTI(GenericOperator);
    base: NeuralNetOperator,
    name: String,
}

impl Default for GenericOperator {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::GenericOperator
        */
    }
}

impl GenericOperator {
    
    pub fn new(name: String) -> Self {
    
        todo!();
        /*
            : NeuralNetOperator(NNKind::GenericOperator), name_(name)
        */
    }
    
    #[inline] pub fn set_name(&mut self, name: String)  {
        
        todo!();
        /*
            name_ = name;
        */
    }
}

impl Named for GenericOperator {

    #[inline] fn get_name(&self) -> String {
        
        todo!();
        /*
            return name_;
        */
    }
}

///--------------------------
pub type NNGraph    = NomGraph<Box<Value>>;
pub type NNSubgraph = Subgraph<Box<Value>>;
pub type NNCFGraph  = ControlFlowGraph<NNGraph>;

///--------------------------
pub struct NNModule<T,U> {
    data_flow:     NNGraph,
    control_flow:  NNCFGraph,
    inputs:        HashSet<NodeRef<T,U>>,
    outputs:       HashSet<NodeRef<T,U>>,
}

impl<T,U> NNModule<T,U> {
    
    /**
      | Simple wrapper of replaceSubgraph where
      | the node is created for you.
      |
      | Returns a NodeRef to the node containing
      | the operator that was created
      */
    #[inline] pub fn replace_subgraph_with_operator<Args>(&mut self, 
        sg:               &SubgraphType<T,U>,
        subgraph_inputs:  &Vec<NodeRef<T,U>>,
        subgraph_outputs: &Vec<NodeRef<T,U>>,
        args:             Args) -> NodeRef<T,U> {

        todo!();
        /*
            auto node = dataFlow.createNode(std::make_unique<T>(args));
      replaceSubgraph(sg, node, subgraph_inputs, subgraph_outputs);
      return node;
        */
    }
    
    #[inline] pub fn create_unique_data_node(
        &mut self, 
        s: Option<&str>) -> NodeRef<T,U> {
        
        let s = s.unwrap_or("_unique");

        todo!();
        /*
            auto curr_name = s;
      auto iter = 0;
      bool need_name = true;
      do {
        need_name = false;
        for (const auto& node : dataFlow.getMutableNodes()) {
          if (nn::getName(node) == curr_name) {
            std::stringstream ss;
            ss << iter;
            curr_name = s + "_" + ss.str();
            iter++;
            need_name = true;
            break;
          }
        }
      } while (need_name);
      return dataFlow.createNode(std::make_unique<Tensor>(curr_name));
        */
    }
    
    /**
      | Replace subgraph sg by node, using the
      | order of node_inputs and node_outputs
      | to determine how to link them to the node.
      | node_inputs *must* enumerate all the
      | inputs to the subgraph (NeuralNetData
      | that do not have producers inside the
      | subgraph). Same for node_outputs
      | 
      | New output names may be created in the
      | case that an inputs and an output have
      | the same name (to avoid in place ops).
      | 
      | This may cause issues with external_output
      | 
      | -- be sure to check after running this
      | function (and perhaps inserting a copy/alias
      | op).
      |
      */
    #[inline] pub fn replace_subgraph(&mut self, 
        subgraph:     &NNSubgraph,
        node:         &NodeRef<T,U>,
        node_inputs:  &Vec<NodeRef<T,U>>,
        node_outputs: &Vec<NodeRef<T,U>>)  {

        todo!();
        /*
            auto sg = subgraph;
      auto sg_inputs = nn::getInputs(sg);
      auto sg_outputs = nn::getOutputs(sg);

      auto sg_inputs_copy = sg_inputs;
      auto sg_outputs_copy = sg_outputs;

      for (const auto& input : node_inputs) {
        sg_inputs_copy.erase(input);
        // outputs may contain inputs that have additional
        // consumers external to the subgraph
        sg_outputs_copy.erase(input);
      }
      assert(sg_inputs_copy.size() == 0 && "Not all inputs were listed");

      for (const auto& output : node_outputs) {
        sg_outputs_copy.erase(output);
      }
      assert(sg_outputs_copy.size() == 0 && "Not all outputs were listed");

      for (auto& input : node_inputs) {
        dataFlow.createEdge(input, node);
        sg.removeNode(input);
      }
      for (auto& output : node_outputs) {
        if (sg_inputs.count(output)) {
          dataFlow.createEdge(node, createUniqueDataNode());
          continue;
        }
        dataFlow.createEdge(node, output);
        sg.removeNode(output);
      }
      deleteSubgraph(sg);
        */
    }
    
    #[inline] pub fn delete_subgraph(&mut self, subgraph: &NNSubgraph)  {
        
        todo!();
        /*
            dataFlow.deleteNodes(subgraph.getNodes());
        */
    }
}

/**
  | Although these seem generic, they make
  | subtle assumptions about the structure
  | of the graph that is 100% valid for NNModule
  | graphs but not any graph (such as data
  | being a unique_ptr).
  |
  */
pub struct inheritedFrom<T,U> {
    phantomA: PhantomData<T>,
    phantomB: PhantomData<U>,
    /*
  static constexpr bool value =
      std::is_base_of<U, T>::value && !std::is_same<U, T>::value;
    */
}

/**
  | This is just a way to fix issues when the
  | isa<> implementation can't automatically
  | downcast.
  |
  */
pub trait is_impl<N> { 

    #[inline] fn impl_(n: N) -> bool {
        
        todo!();
        /*
            return isa<T>(n->data());
        */
    }
}

impl<N> is_impl<N> for NeuralNetOperator {
    
    #[inline] fn impl_(n: N) -> bool {
        
        todo!();
        /*
            if (!isa<NeuralNetOperator>(n->data().get())) {
              return false;
            }
            auto nno = dyn_cast<NeuralNetOperator>(n->data().get());
            return isa<T>(nno);
        */
    }
}

impl<N> is_impl<N> for NeuralNetData {
    
    #[inline] fn impl_(n: N) -> bool {
        todo!();

        /*
            if (!isa<NeuralNetData>(n->data().get())) {
              return false;
            }
            auto nno = dyn_cast<NeuralNetData>(n->data().get());
            return isa<T>(nno);
        */
    }
}

#[inline] pub fn is<T,U>(n: NodeRef<T,U>) -> bool {

    todo!();
    /*
        return is_impl<T, NodeRef>::impl(n);
    */
}

/**
  | This is just a way to fix issues when the
  | dyn_cast<> implementation can't automatically
  | downcast.
  |
  */
pub trait get_impl<N> {
    
    #[inline] fn impl_(n: N) -> *mut Self {
        
        todo!();
        /*
            return dyn_cast<T>(n->data().get());
        */
    }
}

impl<N> get_impl<N> for NeuralNetOperator {
    
    #[inline] fn impl_(n: N) -> *mut Self {

        todo!();
        /*
            if (!is<T>(n)) {
              assert(0 && "Cannot get type from node");
              return nullptr;
            }
            auto nno = dyn_cast<NeuralNetOperator>(n->data().get());
            return dyn_cast<T>(nno);
        */
    }
}

impl<N> get_impl<N> for NeuralNetData {
    
    #[inline] fn impl_(n: N) -> *mut Self {

        todo!();
        /*
            if (!is<T>(n)) {
              assert(0 && "Cannot get type from node");
              return nullptr;
            }
            auto nno = dyn_cast<NeuralNetData>(n->data().get());
            return dyn_cast<T>(nno);
        */
    }
}

#[inline] pub fn get<T, N>(n: N) -> *mut T {

    todo!();
    /*
        return get_impl<T, N>::impl(n);
    */
}

pub fn node_iterator<T, U, G: GraphType>(g: &mut G) -> Vec<<G as GraphType>::NodeRef> {

    todo!();
    /*
      std::vector<typename G::NodeRef> out;
      for (auto node : g.getMutableNodes()) {
        if (!is<T>(node)) {
          continue;
        }
        out.emplace_back(node);
      }
      return out;
    */
}

#[inline] pub fn filter<T, U>(nn: &mut NNModule<T,U>) -> Vec<NodeRef<T,U>> {

    todo!();
    /*
        return nodeIterator<T>(nn.dataFlow);
    */
}

pub fn data_iterator<T, U, G: GraphType>(g: &mut G) -> Vec<(*mut T, <G as GraphType>::NodeRef)> {

    todo!();
    /*
      std::vector<std::pair<T*, typename G::NodeRef>> out;
      for (auto node : g.getMutableNodes()) {
        if (!is<T>(node)) {
          continue;
        }
        auto d = get<T>(node);
        out.emplace_back(std::make_pair(d, node));
      }
      return out;
    */
}

#[inline] pub fn insert_op<T, U, Args>(
    g:    &mut NNGraph,
    a:    NodeRef<T,U>,
    b:    NodeRef<T,U>,
    args: Args)  {

    todo!();
    /*
        if (is<NeuralNetData>(a) && is<NeuralNetOperator>(b)) {
        auto newNode = g.createNode(std::make_unique<T>(args));
        auto data = get<NeuralNetData>(a);
        auto newData =
            g.createNode(std::make_unique<NomTensor>(data->getName() + "_"));
        g.createEdge(a, newNode);
        g.createEdge(newNode, newData);
        g.createEdge(newData, b);
        return;
      }
      if (is<NeuralNetOperator>(a) && is<NeuralNetData>(b)) {
        auto newNode = g.createNode(std::make_unique<T>(args));
        auto data = get<NeuralNetData>(b);
        auto newData =
            g.createNode(std::make_unique<NomTensor>(data->getName() + "_"));
        g.createEdge(a, newData);
        g.createEdge(newData, newNode);
        g.createEdge(newNode, b);
        return;
      }

      assert(0 && "insertOp takes (DFG, NomTensor, Op) or (DFG, Op, NomTensor)");
    */
}

#[inline] pub fn convert_node<NewT,OldT,T,U>(g: &mut NNGraph, node: NodeRef<T,U>) -> NodeRef<T,U> {

    todo!();
    /*
        assert(is<OldT>(node) && "Cannot get type from node.");

      NeuralNetOperator* nnOpPtr =
          dyn_cast<NeuralNetOperator>(node->mutableData()->release());

      auto newNode =
          g.createNode(std::make_unique<NewT>(*dyn_cast<OldT>(nnOpPtr)));

      g.replaceNode(node, newNode);
      g.deleteNode(node);

      return newNode;
    */
}

/// NeuralNetData specific helpers.
#[inline] pub fn has_producer<T,U>(n: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn get_producer<T,U>(n: NodeRef<T,U>) -> NodeRef<T,U> {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn has_consumer<T,U>(n: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn get_consumers<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
    
    */
}


#[inline] pub fn has_inputs<T,U>(n: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn get_inputs_from_node<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn get_outputs_from_node<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
    
    */
}


#[inline] pub fn get_inputs_from_subgraph<T,U>(sg: &NNSubgraph) -> HashSet<NodeRef<T,U>> {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn get_outputs_from_subgraph<T,U>(sg: &NNSubgraph) -> HashSet<NodeRef<T,U>> {
    
    todo!();
    /*
    
    */
}

/**
  | Get the name of the node regardless of
  | underlying type.
  |
  */
#[inline] pub fn get_name<T,U>(n: NodeRef<T,U>) -> String {
    
    todo!();
    /*
    
    */
}

/**
  | Replace the producer of the first argument
  | with the second argument
  |
  */
#[inline] pub fn replace_producer<T,U>(tensor_node: NodeRef<T,U>, new_producer: NodeRef<T,U>)  {
    
    todo!();
    /*
    
    */
}

/**
  | Set all consumers of first argument
  | to consume the second argument
  |
  */
#[inline] pub fn replace_all_uses_with<T,U>(old_tensor_node: NodeRef<T,U>, new_tensor_node: NodeRef<T,U>)  {
    
    todo!();
    /*
    
    */
}

/**
  | Set the second argument to consume the
  | inputs of the first argument
  |
  */
#[inline] pub fn replace_as_consumer<T,U>(old_consumer: NodeRef<T,U>, new_consumer: NodeRef<T,U>)  {
    
    todo!();
    /*
    
    */
}

/// Create an output tensor node
#[inline] pub fn create_output<T,U>(
    nn:       *mut NNModule<T,U>,
    producer: NodeRef<T,U>,
    name:     String) -> NodeRef<T,U> {
    
    todo!();
    /*
    
    */
}

/// Create an operator
#[inline] pub fn create_operator<T, U, Args>(nn: *mut NNModule<T,U>, args: Args) -> NodeRef<T,U> {

    todo!();
    /*
        return nn->dataFlow.createNode(std::make_unique<T>(args));
    */
}

#[inline] pub fn coalesce_inserted_data_dependencies<T,U>(m: *mut NNModule<T,U>)  {
    
    todo!();
    /*
    
    */
}


pub struct NodeHelper {
    g: *mut NNGraph,
}

pub type NNMatchGraph = MatchGraph<NNGraph>;

pub type NNMatchPredicate = MatchPredicate<NNGraph>;

/* ---------- Commonly used node predicate  ---------- */

/**
  | The node has a single output and the output
  | has a single consumer.
  |
  */
#[inline] pub fn has_single_output_and_consumer<T,U>(node_ref: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
    
    */
}

/**
  | The node has a unique consumer (there
  | may be multiple edges from output to
  | the single consumer).
  |
  */
#[inline] pub fn has_unique_consumer<T,U>(node_ref: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn match_external_tensor_node() -> NNMatchPredicate {
    
    todo!();
    /*
    
    */
}
