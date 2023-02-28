crate::ix!();

/**
  | A filler to initialize the parameters
  | and inputs of a predictor
  |
  */
pub trait FillerTrait {

    fn fill_input_internal(&self, input_data: *mut Predictor_TensorList);

    /// initialize the workspace with parameter
    fn fill_parameter(&self, ws: *mut Workspace);
}

pub struct Filler {
    input_names:  Vec<String>,
}

impl Filler {

    /**
      | generate input data and return input
      | data size
      |
      */
    pub fn fill_input(&self, input_data: *mut Predictor_TensorList) -> usize {
        
        todo!();
        /*
            CAFFE_ENFORCE(input_data, "input_data is null");
        input_data->clear();

        fill_input_internal(input_data);

        uint64_t bytes = 0;
        for (const auto& item : *input_data) {
          bytes += item.nbytes();
        }
        if (bytes == 0) {
          LOG(WARNING) << "0 input bytes filled";
        }

        return bytes;
        */
    }

    #[inline] pub fn get_input_names(&self) -> &Vec<String> {
        
        todo!();
        /*
            CAFFE_ENFORCE(!input_names_.empty(), "input names is not initialized");
        return input_names_;
        */
    }
}

/**
  | @init_net: a reader net to generate
  | parameters
  | 
  | @data_net: a reader net to generate
  | inputs
  |
  */
pub struct DataNetFiller {
    base: Filler,

    init_net:  NetDef,
    data_net:  NetDef,
}

impl DataNetFiller {

    pub fn new(init_net: NetDef, data_net: NetDef) -> Self {
    
        todo!();
        /*
            : init_net_(init_net), data_net_(data_net) 

        // The output of the data_net_ will be served as the input
        int op_size = data_net_.op_size();
        for (int i = 0; i < op_size; ++i) {
          OperatorDef op_def = data_net_.op(i);
          // We rely on Fill op to generate inputs
          CAFFE_ENFORCE(op_def.type().find("Fill") != std::string::npos);
          int output_size = op_def.output_size();
          for (int j = 0; j < output_size; ++j) {
            input_names_.push_back(op_def.output(j));
          }
        }
        */
    }

    #[inline] pub fn fill_parameter(&self, ws: *mut Workspace)  {
        
        todo!();
        /*
            // As we use initial parameter initialization for this BenchmarkState,
      // we can just run the init_net
      CAFFE_ENFORCE(
          ws->RunNetOnce(init_net_),
          "Failed running the init_net: ",
          ProtoDebugString(init_net_));
        */
    }
    
    #[inline] pub fn fill_input_internal(&self, input_data: *mut Predictor_TensorList)  {
        
        todo!();
        /*
            Workspace ws;
      CAFFE_ENFORCE(ws.RunNetOnce(data_net_));
      for (const auto& name : input_names_) {
        input_data->emplace_back(
            BlobGetMutableTensor(ws.GetBlob(name), CPU)->Clone());
      }
        */
    }
}

///-------------------------
pub type filler_type_pair_t = (TensorFiller,String);

/**
  | @run_net: the predict net with parameter
  | and input names
  | 
  | @input_dims: the input dimensions
  | of all operator inputs of run_net
  | 
  | @input_types: the input types of all
  | operator inputs of run_net
  |
  */
pub struct DataRandomFiller {
    base: Filler,

    parameters:  HashMap<String,filler_type_pair_t>,
    inputs:      HashMap<String,filler_type_pair_t>,
}

#[inline] pub fn fill_with_type(
    filler: &TensorFiller,
    type_:   &String,
    output: *mut TensorCPU)  {
    
    todo!();
    /*
        CPUContext context;
      if (type == "float") {
        filler.Fill<float>(output, &context);
      } else if (type == "double") {
        filler.Fill<double>(output, &context);
      } else if (type == "uint8_t" || type == "unsigned char") {
        filler.Fill<uint8_t>(output, &context);
      } else if (type == "uint16_t") {
        filler.Fill<uint16_t>(output, &context);
      } else if (type == "int8_t") {
        filler.Fill<int8_t>(output, &context);
      } else if (type == "int16_t") {
        filler.Fill<int16_t>(output, &context);
      } else if (type == "int32_t" || type == "int") {
        filler.Fill<int32_t>(output, &context);
      } else if (type == "int64_t" || type == "long") {
        filler.Fill<int64_t>(output, &context);
      } else if (type == "bool") {
        auto mutable_filler = filler;
        mutable_filler.Min(0).Max(2).Fill<uint8_t>(output, &context);
      } else {
        throw std::invalid_argument("filler does not support type " + type);
      }
    */
}

impl DataRandomFiller {

    #[inline] pub fn get_tensor_filler(
        op_def:      &OperatorDef,
        input_index: i32,
        input_dims:  &Vec<Vec<i64>>) -> TensorFiller {

        todo!();
        /*
            Workspace ws;
        for (int i = 0; i < op_def.input_size(); ++i) {
          // CreateOperator requires all input blobs present
          ws.CreateBlob(op_def.input(i));
        }
        CAFFE_ENFORCE(op_def.has_type());
        const OpSchema* schema = caffe2::OpSchemaRegistry::Schema(op_def.type());
        if (schema == nullptr) {
          throw std::invalid_argument(
              op_def.type() + " does not have input fillers");
        }
        auto filler = schema->InputFillers(input_dims)[input_index];
        return filler;
        */
    }
    
    pub fn new(
        run_net:     &NetDef,
        input_dims:  &Vec<Vec<Vec<i64>>>,
        input_types: &Vec<Vec<String>>) -> Self {
    
        todo!();
        /*
            // parse dimensions
      CAFFE_ENFORCE_EQ(input_dims.size(), run_net.op_size());
      CAFFE_ENFORCE_EQ(input_types.size(), run_net.op_size());

      // load op inputs and outputs
      std::unordered_set<std::string> output_names;
      for (size_t i = 0; i < run_net.op_size(); ++i) {
        const auto& op = run_net.op(i);
        const auto& op_dims = input_dims[i];
        const auto& op_types = input_types[i];
        CAFFE_ENFORCE(
            op_dims.size() == op.input_size(),
            op.name() + " has " + c10::to_string(op.input_size()) +
                " inputs; while the input dimension size is " +
                c10::to_string(op_dims.size()));
        CAFFE_ENFORCE(
            op_types.size() == op.input_size(),
            op.name() + " has " + c10::to_string(op.input_size()) +
                " inputs; while the input type size is " +
                c10::to_string(op_types.size()));

        for (size_t j = 0; j < op.input_size(); ++j) {
          inputs_[op.input(j)] =
              std::make_pair(get_tensor_filler(op, j, op_dims), op_types[j]);
        }

        // Hack, we normal have a path of
        // length -> LengthsiRangeFill -> Gather -> w -> SparseLengthsWeighted*
        //       \---------------------------------------/
        // So when we generate the value of length, we need to bound it to the size
        // of weight input of Gather too
        if (op.type().find("SparseLengthsWeighted") == 0 && i > 0) {
          const auto& prev_op = run_net.op(i - 1);
          if (prev_op.type() == "Gather") {
            const auto& prev_dims = input_dims[i - 1];
            VLOG(1) << "Setting max length value to " << prev_dims[0].front()
                    << " for " << op.input(3);
            inputs_[op.input(3)].first.Max(prev_dims[0].front());
          }
        }

        for (size_t j = 0; j < op.output_size(); ++j) {
          output_names.emplace(op.output(j));
        }
      }

      // load parameters
      std::unordered_set<std::string> parameters;
      for (size_t i = 0; i < run_net.arg_size(); ++i) {
        const auto& arg = run_net.arg(i);
        // TODO: replace "PredictorParameters" with the constant in OSS bbp
        if (arg.has_name() && arg.name() == "PredictorParameters") {
          parameters.reserve(arg.strings_size());
          for (size_t j = 0; j < arg.strings_size(); ++j) {
            parameters.emplace(arg.strings(j));
          }
          break;
        }
      }
      if (parameters.size() == 0) {
        VLOG(1) << "Fail to find any parameters";
      }
      for (const auto& param : parameters) {
        // remove unused parameters
        if (inputs_.find(param) != inputs_.end()) {
          // inputs_[param] will be erase from inputs_ in the next step
          parameters_.emplace(param, inputs_[param]);
        }
      }

      for (const auto& param : parameters_) {
        inputs_.erase(param.first);
      }
      for (const auto& name : output_names) {
        inputs_.erase(name);
      }
      CAFFE_ENFORCE(inputs_.size() > 0, "Empty input for run net");

      // generate input names
      for (const auto& input : inputs_) {
        input_names_.push_back(input.first);
      }
        */
    }
    
    #[inline] pub fn fill_parameter(&self, ws: *mut Workspace)  {
        
        todo!();
        /*
            for (auto& param : parameters_) {
        Blob* blob = ws->CreateBlob(param.first);
        fill_with_type(
            param.second.first,
            param.second.second,
            BlobGetMutableTensor(blob, CPU));
        CAFFE_ENFORCE(ws->GetBlob(param.first)->GetRaw());
      }
        */
    }
    
    #[inline] pub fn fill_input_internal(&self, input_data: *mut Predictor_TensorList)  {
        
        todo!();
        /*
            for (auto& name : input_names_) {
        input_data->emplace_back(CPU);
        const auto& it = inputs_.find(name);
        CAFFE_ENFORCE(it != inputs_.end());
        fill_with_type(it->second.first, it->second.second, &input_data->back());
      }
        */
    }
    
}

///-------------------------------
/**
  A DataRandomFiller that is more convenient to use in unit tests.
  Callers just need to supply input dimensions and types for non-intermediate
  blobs.
  It also treats parameters the same way as non-intermediate inputs (no
  handling of parameters separately).
  */
pub struct TestDataRandomFiller {
    base: DataRandomFiller,
}

impl TestDataRandomFiller {
    
    pub fn new(
        net:         &NetDef,
        input_dims:  &Vec<Vec<Vec<i64>>>,
        input_types: &Vec<Vec<String>>) -> Self {

        todo!();
        /*
            : DataRandomFiller() 

      std::unordered_set<std::string> outputNames;
      // Determine blobs that are outputs of some ops (intermediate blobs).
      for (auto opIdx = 0; opIdx < net.op_size(); ++opIdx) {
        const auto& op = net.op(opIdx);
        for (auto outputIdx = 0; outputIdx < op.output_size(); ++outputIdx) {
          outputNames.emplace(op.output(outputIdx));
        }
      }
      // Determine ops that have non-intermediate inputs.
      std::unordered_set<size_t> opWithRequiredInputs;
      for (auto opIdx = 0; opIdx < net.op_size(); ++opIdx) {
        const auto& op = net.op(opIdx);
        for (auto inputIdx = 0; inputIdx < op.input_size(); ++inputIdx) {
          if (!outputNames.count(op.input(inputIdx))) {
            opWithRequiredInputs.emplace(opIdx);
            break;
          }
        }
      }

      CAFFE_ENFORCE_EQ(inputDims.size(), opWithRequiredInputs.size());
      CAFFE_ENFORCE_EQ(inputTypes.size(), opWithRequiredInputs.size());

      int counter = 0;
      for (auto opIdx = 0; opIdx < net.op_size(); ++opIdx) {
        if (!opWithRequiredInputs.count(opIdx)) {
          // Skip intermediate ops.
          continue;
        }
        const auto& op = net.op(opIdx);
        const auto& op_dims = inputDims[counter];
        const auto& op_types = inputTypes[counter];
        ++counter;

        int countRequiredInputs = 0;
        for (auto inputIdx = 0; inputIdx < op.input_size(); ++inputIdx) {
          if (!outputNames.count(op.input(inputIdx))) {
            ++countRequiredInputs;
          }
        }

        CAFFE_ENFORCE(
            op_dims.size() == countRequiredInputs,
            op.name() + " has " + c10::to_string(op.input_size()) +
                " (required) inputs; while the input dimension size is " +
                c10::to_string(op_dims.size()));
        CAFFE_ENFORCE(
            op_types.size() == countRequiredInputs,
            op.name() + " has " + c10::to_string(op.input_size()) +
                " (required) inputs; while the input type size is " +
                c10::to_string(op_types.size()));

        int dimCounter = 0;
        for (auto inputIdx = 0; inputIdx < op.input_size(); ++inputIdx) {
          auto inputName = op.input(inputIdx);
          if (outputNames.count(inputName)) {
            // Skip intermediate inputs.
            continue;
          }
          inputs_[inputName] = std::make_pair(
              get_tensor_filler(op, dimCounter, op_dims), op_types[dimCounter]);
          ++dimCounter;
        }
      }
      CAFFE_ENFORCE(inputs_.size() > 0, "Empty input for run net");
      // generate input names
      for (const auto& input : inputs_) {
        input_names_.push_back(input.first);
      }
        */
    }
    
    /// Fill input directly to the workspace.
    #[inline] pub fn fill_input_to_workspace(&self, workspace: *mut Workspace)  {
        
        todo!();
        /*
            for (auto& name : input_names_) {
        const auto& it = inputs_.find(name);
        CAFFE_ENFORCE(it != inputs_.end());
        auto* tensor =
            BlobGetMutableTensor(workspace->CreateBlob(name), caffe2::CPU);
        fill_with_type(it->second.first, it->second.second, tensor);
      }
        */
    }
}

/// Convenient helpers to fill data to workspace.
#[inline] pub fn fill_random_network_inputs(
    net:         &NetDef,
    input_dims:  &Vec<Vec<Vec<i64>>>,
    input_types: &Vec<Vec<String>>,
    workspace:   *mut Workspace)  {

    todo!();
    /*
        TestDataRandomFiller(net, inputDims, inputTypes)
          .fillInputToWorkspace(workspace);
    */
}
