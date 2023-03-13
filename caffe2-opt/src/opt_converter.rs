crate::ix!();



///---------------------
pub trait Converter {

    fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator>;

    #[inline] fn get_arguments_from_operator(&self, op: OperatorDef) -> HashMap<String,Argument> {
        
        todo!();
        /*
            std::map<std::string, caffe2::Argument> argMap;
      for (auto arg : op.arg()) {
        argMap[arg.name()] = arg;
      }
      return argMap;
        */
    }

    #[inline] fn convert_to_operator_def(&mut self, nn_op: *const NeuralNetOperator) -> OperatorDef {
        
        todo!();
        /*
            auto* annotation = nnOp->getAnnotation();
      // Default to using the stored operator.
      if (annotation && isa<Caffe2Annotation>(annotation)) {
        return dyn_cast<Caffe2Annotation>(annotation)->getOperatorDef();
      }
      LOG(WARNING)
          << "Cannot instantiate this OperatorDef from nomnigraph, falling back";
      caffe2::OperatorDef op;
      op.set_type(nnOp->getName());
      return op;
        */
    }
    
    #[inline] fn get_device_option(&self, nn_op: *const NeuralNetOperator) -> DeviceOption {
        
        todo!();
        /*
            auto* annotation = nnOp->getAnnotation();
      // Default to using the stored operator.
      if (annotation && isa<Caffe2Annotation>(annotation)) {
        return dyn_cast<Caffe2Annotation>(annotation)
            ->getOperatorDef()
            .device_option();
      }
      caffe2::DeviceOption opt;
      return opt;
        */
    }
    
    #[inline] fn get_kernel_shape(&mut self, arg_map: HashMap<String,Argument>) -> Vec<i32> {
        
        todo!();
        /*
            // There are literally three ways to define shapes in Conv in Caffe2
      std::vector<int> kernelShape;
      if (argMap.count("kernel")) {
        CAFFE_ENFORCE(argMap["kernel"].has_i(), "Invalid kernel argument");
        int kernel = static_cast<int>(argMap["kernel"].i());
        kernelShape = {kernel, kernel};
      } else if (argMap.count("kernels")) {
        for (auto i : argMap["kernels"].ints()) {
          kernelShape.push_back(static_cast<int>(i));
        }
      } else if (argMap.count("kernel_h") && argMap.count("kernel_w")) {
        CAFFE_ENFORCE(argMap["kernel_h"].has_i(), "Invalid kernel argument");
        CAFFE_ENFORCE(argMap["kernel_w"].has_i(), "Invalid kernel argument");
        int kernelH = static_cast<int>(argMap["kernel_h"].i());
        int kernelW = static_cast<int>(argMap["kernel_w"].i());
        kernelShape = {kernelH, kernelW};
      }
      return kernelShape;
        */
    }
}

declare_registry!{
    ConverterRegistry, 
    Converter
}

#[inline] pub fn get_strides(arg_map: HashMap<String,Argument>) -> Vec<i32> {
    
    todo!();
    /*
        std::vector<int> strides;
      // TODO: include all the other ways of adding these args.
      // e.g. strides, stride_h, etc.
      if (argMap.count("stride")) {
        CAFFE_ENFORCE(argMap["stride"].has_i(), "Invalid stride argument");
        int stride = static_cast<int>(argMap["stride"].i());
        strides = {stride, stride};
      }
      return strides;
    */
}

#[inline] pub fn get_pads(arg_map: HashMap<String,Argument>) -> Vec<i32> {
    
    todo!();
    /*
        std::vector<int> pads;
      if (argMap.count("pad")) {
        CAFFE_ENFORCE(argMap["pad"].has_i(), "Invalid pad argument");
        int pad = static_cast<int>(argMap["pad"].i());
        pads = {pad, pad, pad, pad};
      }
      return pads;
    */
}

#[inline] pub fn get_dilations(arg_map: HashMap<String,Argument>) -> Vec<i32> {
    
    todo!();
    /*
        std::vector<int> dilations;
      if (argMap.count("dilation")) {
        CAFFE_ENFORCE(argMap["dilation"].has_i(), "Invalid dilation argument");
        int dilation = static_cast<int>(argMap["dilation"].i());
        dilations = {dilation, dilation};
      }
      return dilations;
    */
}

#[inline] pub fn get_group(arg_map: &mut HashMap<String,Argument>) -> i32 {
    
    todo!();
    /*
        if (argMap.count("group")) {
        CAFFE_ENFORCE(argMap["group"].has_i() && "Invalid group argument");
        return static_cast<int>(argMap["group"].i());
      }
      return 1;
    */
}

define_registry!{/*ConverterRegistry, Converter*/}

#[inline] pub fn get_layout(arg_map: HashMap<String,Argument>)  {
    
    todo!();
    /*
        auto arg = argMap.find("order");
      if (arg != argMap.end()) {
        auto order = argMap["order"].s();
        if (order == "NCHW" || order == "nchw") {
          return repr::NeuralNetOperator::NNLayout::NCHW;
        } else if (order == "NHWC" || order == "nhwc") {
          return repr::NeuralNetOperator::NNLayout::NHWC;
        }
      }
      return repr::NeuralNetOperator::NNLayout::Undefined;
    */
}

///-----------------------------
pub struct ConvConverter {
    base: dyn Converter,
}

impl ConvConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp;
        auto argMap = getArgumentsFromOperator(op);
        auto kernelShape = getKernelShape(argMap);
        nnOp = std::make_unique<repr::Conv>(kernelShape);
        auto c = dyn_cast<repr::Conv>(nnOp.get());

        c->setStrides(getStrides(argMap));
        c->setPads(getPads(argMap));
        c->setDilations(getDilations(argMap));
        c->setGroup(getGroup(argMap));

        return nnOp;
        */
    }
}

///------------------------------
pub struct ConvTransposeConverter {
    base: dyn Converter,
}

impl ConvTransposeConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp;
        auto argMap = getArgumentsFromOperator(op);
        auto kernelShape = getKernelShape(argMap);
        nnOp = std::make_unique<repr::ConvTranspose>(kernelShape);
        auto c = dyn_cast<repr::ConvTranspose>(nnOp.get());

        c->setStrides(getStrides(argMap));
        c->setPads(getPads(argMap));
        c->setGroup(getGroup(argMap));

        return nnOp;
        */
    }
}

register_converter!{Conv, ConvConverter}

register_converter!{ConvTranspose, ConvTransposeConverter}

trivial_converter!{Relu}
register_converter!{Relu, ReluConverter}

trivial_converter!{Sum}
register_converter!{Sum, SumConverter}

trivial_converter!{BatchNormalization}
register_converter!{SpatialBN, BatchNormalizationConverter}

trivial_converter!{Flatten}
register_converter!{Flatten, FlattenConverter}

///----------------------------------
pub struct ClipConverter {
    base: dyn Converter,
}

impl ClipConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            auto argMap = getArgumentsFromOperator(op);
        float min = std::numeric_limits<float>::lowest();
        float max = float::max;

        if (argMap.count("min")) {
          CAFFE_ENFORCE(argMap["min"].has_f(), "Invalid 'min' argument");
          min = static_cast<float>(argMap["min"].f());
        }

        if (argMap.count("max")) {
          CAFFE_ENFORCE(argMap["max"].has_f(), "Invalid 'max' argument");
          max = static_cast<float>(argMap["max"].f());
        }

        return std::make_unique<repr::Clip>(min, max);
        */
    }
}

register_converter!{Clip, ClipConverter}

///---------------------------------------
pub struct AveragePoolConverter {
    base: dyn Converter,
}

impl AveragePoolConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp;
        auto argMap = getArgumentsFromOperator(op);
        auto kernelShape = getKernelShape(argMap);
        nnOp = std::make_unique<repr::AveragePool>(kernelShape);
        return nnOp;
        */
    }
}

register_converter!{AveragePool, AveragePoolConverter}

///------------------------------------
pub struct MaxPoolConverter {
    base: dyn Converter,
}
impl MaxPoolConverter {

    /// Does not override default converter to OperatorDef
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp;
        auto argMap = getArgumentsFromOperator(op);
        auto kernelShape = getKernelShape(argMap);
        nnOp = std::make_unique<repr::MaxPool>(kernelShape);
        return nnOp;
        */
    }
}

register_converter!{MaxPool, MaxPoolConverter}

///--------------------------------------------
pub struct ConcatConverter {
    base: dyn Converter,
}

impl ConcatConverter {

    /**
      | Does not override default converter
      | to OperatorDef
      |
      */
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::Concat>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::Concat>(nnOp.get());
        if (argMap.count("axis")) {
          CAFFE_ENFORCE(argMap["axis"].has_i(), "Invalid axis argument");
          int axis = static_cast<int>(argMap["axis"].i());
          c->setAxis(axis);
        }
        if (argMap.count("add_axis")) {
          CAFFE_ENFORCE(argMap["add_axis"].has_i(), "Invalid add_axis argument");
          int add_axis = static_cast<int>(argMap["add_axis"].i());
          c->setAddAxis(!!add_axis);
        }
        return nnOp;
        */
    }
}

register_converter!{Concat, ConcatConverter}

///---------------------------
pub struct FCConverter {
    base: dyn Converter,
}

impl FCConverter {
    
    /**
      | Does not override default converter
      | to OperatorDef
      |
      */
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::FC>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::FC>(nnOp.get());
        if (argMap.count("axis")) {
          CAFFE_ENFORCE(argMap["axis"].has_i(), "Invalid axis argument");
          int axis = static_cast<int>(argMap["axis"].i());
          c->setAxis(axis);
        }
        if (argMap.count("axis_w")) {
          CAFFE_ENFORCE(argMap["axis_w"].has_i(), "Invalid axis_w argument");
          int axis_w = static_cast<int>(argMap["axis_w"].i());
          c->setAxisW(axis_w);
        }

        return nnOp;
        */
    }
}

register_converter!{FC, FCConverter}

///--------------------------------

/**
  | Use these functions instead of the registry
  | directly.
  |
  */
#[inline] pub fn convert_to_neural_net_operator(op: &OperatorDef) -> Box<NeuralNetOperator> {
    
    todo!();
    /*
        auto argMap = Converter::getArgumentsFromOperator(op);

      std::unique_ptr<repr::NeuralNetOperator> nnOp;

      if (ConverterRegistry()->Has(op.type())) {
        nnOp =
            ConverterRegistry()->Create(op.type())->convertToNeuralNetOperator(op);
      }

      if (!nnOp) {
        nnOp = std::make_unique<repr::GenericOperator>(op.type());
      }

      // Generic attributes associated with Ops here
      nnOp->setLayout(getLayout(argMap));

      auto annotation = std::make_unique<Caffe2Annotation>();
      annotation->setOperatorDef(op);

      auto device_name = op.device_option().node_name();
      if (device_name != "") {
        annotation->setDevice(device_name);
      }
      annotation->setDeviceType(op.device_option().device_type());

      nnOp->setAnnotation(std::move(annotation));

      return nnOp;
    */
}

/**
  | \brief Ingest a caffe2 protobuf model and
  | output an NNModule.
  |
  | \param net The caffe2 protobuf NetDef
  |
  | Default conversion to a NNModule
  |
  | Optionally strict -- which checks for various
  | input and output conditions.
  |
  | Optionally this function will update a vector
  | that maps operators in the netdef positionally
  | to NodeRefs in the resultant NNModule.
  */
#[inline] pub fn convert_to_nnmodule<T,U>(
    net:         &NetDef,
    strict:      Option<bool>,
    op_node_vec: *mut Vec<NodeRef<T,U>>) -> NNModule<T,U> 
{
    let strict = strict.unwrap_or(false);
    
    todo!();
    /*
        repr::NNModule module;
      repr::NNGraph& dfg = module.dataFlow;
      repr::NNCFGraph& cfg = module.controlFlow;
      /// \brief We keep track of the producer of the blob.
      /// Because Caffe2 Nets are really just ordered operations
      /// we can just keep track of the most recent producer of
      /// a blob and draw and edge from that to any consumer we
      /// come by. If a new operator produces the blob we simply
      /// replace it in this map.
      std::unordered_map<std::string, repr::NNGraph::NodeRef> blobMap;

      std::unordered_set<std::string> externalInputNames;
      for (const auto& inputName : net.external_input()) {
        externalInputNames.insert(inputName);
      }

      /// \brief For the construction of the control flow graph we keep track
      /// of a current basic block, which we split up as we come across control
      /// flow operations such as if and while.
      auto bbNode = cfg.createNamedFunction("main");

      for (const auto& op : net.op()) {
        auto opNode = dfg.createNode(); // Create an empty node for the operator.
        // First calculate in-edges (data dependencies).
        for (const auto& input : op.input()) {
          // If we've never seen this tensor, make one.
          if (!blobMap.count(input)) {
            auto tensor = std::make_unique<repr::Tensor>(input);
            blobMap[input] =
                dfg.createNode(unique_dyn_cast<repr::NeuralNetData>(tensor));
            if (externalInputNames.count(input)) {
              module.inputs.insert(blobMap[input]);
              externalInputNames.erase(input);
            }
          }

          auto tensorNode = blobMap[input];
          dfg.createEdge(tensorNode, opNode);
        }

        // Then save outputs into the blobMap for later consumption.
        for (const auto& output : op.output()) {
          auto tensor = std::make_unique<repr::Tensor>(output);
          auto tensorNode =
              dfg.createNode(unique_dyn_cast<repr::NeuralNetData>(tensor));
          dfg.createEdge(opNode, tensorNode);
          blobMap[output] = tensorNode;
        }

        opNode->resetData(convertToNeuralNetOperator(op));
        if (opNodeVec) {
          opNodeVec->emplace_back(opNode);
        }
        auto currentBasicBlock = bbNode->mutableData();
        currentBasicBlock->pushInstructionNode(opNode);
      }

      if (externalInputNames.size()) {
        // In strict mode we ensure the input names are valid
        if (strict) {
          std::ostringstream os;
          for (const auto& inputName : externalInputNames) {
            os << "\"" << inputName << "\" ";
          }

          CAFFE_ENFORCE(
              externalInputNames.size() == 0,
              "Attempting to convert an ill-formed network: ",
              "external_input contains ",
              externalInputNames.size(),
              " unused blobs: ",
              os.str());
          // Otherwise, we add the blobs to the graph as no-ops
        } else {
          for (const auto& input : externalInputNames) {
            blobMap[input] = dfg.createNode(std::make_unique<repr::Tensor>(input));
          }
        }
      }

      for (const auto& outputName : net.external_output()) {
        CAFFE_ENFORCE(
            !strict || blobMap.count(outputName),
            "NetDef has ill-formed external_output:",
            outputName);
        if (!blobMap.count(outputName)) {
          LOG(ERROR) << "NetDef has ill-formed external_output: " << outputName;
          continue;
        }
        module.outputs.insert(blobMap[outputName]);
      }

      return module;
    */
}

#[inline] pub fn convert_to_operator_def<T,U>(instr_node: &NodeRef<T,U>) -> OperatorDef {
    
    todo!();
    /*
        auto* nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
      auto op_type = nnOp->getName();
      auto* annotation = nnOp->getAnnotation();
      caffe2::OperatorDef op;

      if (ConverterRegistry()->Has(op_type)) {
        op = ConverterRegistry()->Create(op_type)->convertToOperatorDef(nnOp);
      } else if (!annotation) {
        op.set_type(op_type);
      } else {
        if (isa<Caffe2Annotation>(annotation)) {
          auto c2_annotation = dyn_cast<Caffe2Annotation>(annotation);
          op = c2_annotation->getOperatorDef();
          op.mutable_device_option()->set_device_type(
              c2_annotation->getDeviceType());
        } else {
          CAFFE_THROW(
              "Couldn't convert operator annotation to Caffe2 operator def");
        }
      }

      // We may have swapped out some of the edges.
      op.clear_input();
      op.clear_output();
      return op;
    */
}

/**
  | If the annotation doesn't exist, attempt
  | to add it
  |
  */
#[inline] pub fn get_or_add_caffe2_annotation<T,U>(instr_node: &mut NodeRef<T,U>) -> *mut Caffe2Annotation<T,U> {
    
    todo!();
    /*
        auto* nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
      auto* annotation = nnOp->getMutableAnnotation();
      if (!annotation) {
        auto new_annot = std::make_unique<Caffe2Annotation>();
        new_annot->setOperatorDef(convertToOperatorDef(instrNode));
        nnOp->setAnnotation(std::move(new_annot));
        annotation = nnOp->getMutableAnnotation();
      }
      CAFFE_ENFORCE(isa<Caffe2Annotation>(annotation));
      auto c2_annotation = dyn_cast<Caffe2Annotation>(annotation);
      return c2_annotation;
    */
}

#[inline] pub fn convert_to_caffe_2proto<T,U>(m: &mut NNModule<T,U>) -> NetDef {
    
    todo!();
    /*
        auto predictNet = caffe2::NetDef();
      return convertToCaffe2Proto(m, predictNet);
    */
}


#[inline] pub fn merge_external_tensors<T,U>(
    curr_external: &HashSet<NodeRef<T,U>>, 
    old_external: &Vec<String>) -> Vec<String> {

    todo!();
    /*
        std::vector<std::string> out;

      // Maximally preserve the order of external inputs and outputs.
      std::unordered_set<std::string> newExternal;
      for (const auto& tensorNode : currExternal) {
        CAFFE_ENFORCE(
            repr::nn::is<repr::NeuralNetData>(tensorNode),
            "A non-tensor node was added to external inputs/outputs of the NNModule");
        auto name = repr::nn::get<repr::NeuralNetData>(tensorNode)->getName();
        newExternal.insert(name);
      }

      for (const auto& tensorName : oldExternal) {
        if (newExternal.count(tensorName)) {
          out.emplace_back(tensorName);
          newExternal.erase(tensorName);
        }
      }
      for (const auto& tensorName : newExternal) {
        out.emplace_back(tensorName);
      }

      return out;
    */
}

/**
  | Pass in an oldNet to copy all the attributes
  | of that network.
  |
  | Be warned that transformations that modify the
  | graph's inputs or outputs are not reflected in
  | changes to external_input or external_output.
  */
#[inline] pub fn convert_to_caffe_2proto_with_old_net<T,U>(
    m: &mut NNModule<T,U>, 
    old_net: &NetDef) -> NetDef 
{
    todo!();
    /*
        auto predictNet = caffe2::NetDef();
      // We copy the old net rather than mutate it.
      predictNet.CopyFrom(oldNet);
      predictNet.mutable_op()->Clear();

      repr::nn::coalesceInsertedDataDependencies(&m);

      // Simply iterate through the CFG and populate data dependencies
      // with the DFG
      for (const auto& bbNode : m.controlFlow.getMutableNodes()) {
        if (bbNode->getOutEdges().size() > 1) {
          CAFFE_THROW("Control flow not yet supported in Caffe2 converter.");
        }
        auto& bb = bbNode->data();
        for (const auto& instrNode : bb.getInstructions()) {
          caffe2::OperatorDef op = convertToOperatorDef(instrNode);

          for (const auto& inEdge : instrNode->getInEdges()) {
            auto* tensorNode =
                dyn_cast<repr::NeuralNetData>(inEdge->tail()->data().get());
            *op.add_input() = tensorNode->getName();
          }
          for (const auto& outEdge : instrNode->getOutEdges()) {
            auto* tensorNode =
                dyn_cast<repr::NeuralNetData>(outEdge->head()->data().get());
            *op.add_output() = tensorNode->getName();
          }

          auto* nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
          if (nnOp->getLayout() != repr::NeuralNetOperator::NNLayout::Undefined) {
            caffe2::Argument* arg = nullptr;
            for (int i = 0; i < op.arg_size(); ++i) {
              auto arg_ = op.mutable_arg(i);
              if (arg_->name() == "order") {
                arg = arg_;
                break;
              }
            }

            if (!arg) {
              arg = op.add_arg();
              arg->set_name("order");
            }

            auto layout = nnOp->getLayout();
            if (layout == repr::NeuralNetOperator::NNLayout::NCHW) {
              arg->set_s("NCHW");
            }
            if (layout == repr::NeuralNetOperator::NNLayout::NHWC) {
              arg->set_s("NHWC");
            }
          }

          // Save the operator to the net.
          *predictNet.add_op() = op;
        }
      }

      // Maximally preserve the order of external inputs and outputs.
      std::vector<std::string> oldExternalInputs;
      std::vector<std::string> oldExternalOutputs;

      for (const auto& inputName : predictNet.external_input()) {
        oldExternalInputs.emplace_back(inputName);
      }
      for (const auto& outputName : predictNet.external_output()) {
        oldExternalOutputs.emplace_back(outputName);
      }

      auto newExternalInputs = mergeExternalTensors(m.inputs, oldExternalInputs);
      auto newExternalOutputs = mergeExternalTensors(m.outputs, oldExternalOutputs);

      predictNet.clear_external_input();
      predictNet.clear_external_output();

      for (const auto& inputName : newExternalInputs) {
        predictNet.add_external_input(inputName);
      }

      for (const auto& outputName : newExternalOutputs) {
        predictNet.add_external_output(outputName);
      }

      return predictNet;
    */
}

#[inline] pub fn push_op_to_front(op: &mut OperatorDef, net: *mut NetDef)  {
    
    todo!();
    /*
        *net->add_op() = op;
      google::protobuf::RepeatedPtrField<caffe2::OperatorDef>* op_list(
          net->mutable_op());
      // Reverse iterate, swapping new element in front each time
      for (int i(net->op_size() - 1); i > 0; --i) {
        op_list->SwapElements(i, i - 1);
      }
    */
}

#[inline] pub fn inject_data_edge_indicators(net: *mut NetDef)  {
    
    todo!();
    /*
        for (const auto& input : net->external_input()) {
        caffe2::OperatorDef op;
        op.set_type("Declare");
        op.add_output(input);
        pushOpToFront(op, net);
      }
      for (const auto& output : net->external_output()) {
        caffe2::OperatorDef op;
        op.set_type("Export");
        op.add_input(output);
        *net->add_op() = std::move(op);
      }
      net->clear_external_input();
      net->clear_external_output();
    */
}

#[inline] pub fn remove_data_edge_indicators(net: *mut NetDef)  {
    
    todo!();
    /*
        google::protobuf::RepeatedPtrField<caffe2::OperatorDef>* op_list(
          net->mutable_op());
      for (auto i = 0; i < net->op_size(); ++i) {
        auto op = net->op(i);
        if (op.type() == "Declare") {
          net->add_external_input(op.output(0));
        } else if (op.type() == "Export") {
          net->add_external_output(op.input(0));
        } else {
          continue;
        }
        // Note that this compensates for modifying the list inplace
        op_list->DeleteSubrange(i--, 1);
      }
    */
}
