crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct OnnxifiOp<Context> {

    storage: OperatorStorage,
    context: Context,

    /// pointer to loaded onnxifi library
    lib:                       *mut OnnxifiLibrary, // default = nullptr

    backend_graph_map_ptr:     *mut OnnxBackendGraphMap,
    op_id_string:              String,
    backend_id:                OnnxBackendID, // default = nullptr
    backend:                   OnnxBackend, // default = nullptr
    graph:                     OnnxGraph,   // default = nullptr
    backend_graph_shared_ptr:  OnnxSharedPtrBackendGraphInfo,

    /// input/output descriptors
    input_desc:                Vec<OnnxTensorDescriptorV1>,

    output_desc:               Vec<OnnxTensorDescriptorV1>,

    /**
      | Output reshape info
      |
      | It is a map keyed on batch size and the
      | value OutputReshapeInfo for the batch
      | size.
      */
    output_reshape_info:       HashMap<i32,OutputReshapeInfo>,

    /// onnxifi extension mode function pointer
    #[cfg(onnxifi_enable_ext)]
    onnx_set_io_and_run_graph_pointer: fn(
        OnnxGraph, 
        u32, 
        *const OnnxTensorDescriptorV1, 
        u32, 
        *const OnnxTensorDescriptorV1, 
        *mut OnnxMemoryFenceV1,
        *mut OnnxTraceEventList) -> OnnxStatus,

    #[cfg(onnxifi_enable_ext)]
    onnx_release_trace_events_pointer: fn(
        *mut OnnxTraceEventList) -> OnnxStatus,

    #[cfg(onnxifi_enable_ext)]
    onnx_wait_event_for_pointer: fn(
        event:          OnnxEvent,
        timeout_ms:     u32,
        event_state:    *mut OnnxEventState,
        message:        *mut u8,
        message_length: *mut usize) -> OnnxStatus,

    #[cfg(onnxifi_enable_ext)]
    traces:  Arc<OnnxTraceEventList>, // default = nullptr

    /// ONNX model or not
    use_onnx:                  bool, // default = false

    /// Glow AOT model or not
    use_glow_aot:              bool, // default = false

    /// max batch size
    max_batch_size:            i32,

    /// max sequence lookup size
    max_seq_size:              i32,

    /**
      | Inference timeout limits. Default
      | 0 means no timeout.
      |
      */
    timeout:                   i32,

    /**
      | index of the input whose first dimension
      | represents the batch size
      |
      */
    nominal_batch_idx:         i32, // default = 0

    /**
      | We bind the op input/output by position
      | while ONNXIFI binds input/output by
      | names. In addition, op input/output names
      | can be written by, for example,
      | memonger. We cache the original
      | input/output name of ONNX object here and
      | bind them by position.
      */
    input_names:               Vec<String>,

    output_names:              Vec<String>,

    /**
      | NetDef of the onnxifi subgraph for shape
      | inference
      |
      */
    netdef:                    NetDef,

    input_shapes:              Vec<SmallVec<[u64; 4]>>,
    output_shapes_max_bs:      Vec<SmallVec<[u64; 4]>>,

    /// Mapping of batch sizes to output shapes
    output_shapes_per_bs:      HashMap<i32, Vec<SmallVec<[u64; 4]>>>,

    /// Indicate if i-th output is a quantized tensor
    quantized_outputs:         Vec<bool>,

    /// This is for multi group quantization info
    all_scales:                Vec<Vec<f32>>,
    all_offsets:               Vec<Vec<i32>>,

    /// output shape hints
    output_shape_hints:        HashMap<i32,TensorInfo>,

    /**
      | input shape info. Used by shape inference
      | when inputs are not at max_batch_size
      |
      */
    input_shape_info:          HashMap<String,ShapeInfo>,

    /**
      | Whether we should use passed output
      | shape hints or do shape inference
      |
      */
    use_passed_output_shapes:  bool, // default = false

    /// Whether we need to resize outputs or not
    adjust_output_batch:       bool, // default = false

    /// Whether we enable tracing in one run of inference
    enable_tracing:            bool, // default = false

    /**
      | Adjust the quantized offset to compensate
      | mismatch of certain backend
      |
      */
    adjust_quantized_offset:   u8, // default = 0
}

/**
  | The Onnxifi operator is a black-box
  | operator to lower the computation to
  | Onnxifi backend
  |
  */
register_cpu_operator!{Onnxifi, OnnxifiOp<CPUContext>}

num_inputs!{Onnxifi, (0,INT_MAX)}

num_outputs!{Onnxifi, (0,INT_MAX)}

args!{Onnxifi, 
    0 => ("onnx_model", "(string default='') Serialized ONNX model to be converted to backend representation"),
    1 => ("initializers", "Initialization pair indicating the mapping of the name between NetDef and ONNX model"),
    2 => ("output_resize_hints", "A list of key/value pairs indicating which input index to look up for real batch size for the given max output batch size")
}


impl<Context> Drop for OnnxifiOp<Context> {
    fn drop(&mut self) {
        todo!();
        /* 
        backend_graph_shared_ptr_.reset();
        backend_graph_map_ptr_->remove(op_id_string_);
    #ifdef ONNXIFI_ENABLE_EXT
        traces_.reset();
    #endif
       */
    }
}

impl<Context> OnnxifiOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            use_onnx_(this->template GetSingleArgument<int>("use_onnx", 0)),
            use_glow_aot_(this->template GetSingleArgument<int>("use_glow_aot", 0)),
            max_batch_size_( this->template GetSingleArgument<int>("max_batch_size", 0)),
            max_seq_size_(this->template GetSingleArgument<int>("max_seq_size", 0)),
            timeout_(this->template GetSingleArgument<int>("timeout", 0)),
            nominal_batch_idx_( this->template GetSingleArgument<int>("nominal_batch_idx", 0)),
            use_passed_output_shapes_(this->template GetSingleArgument<int>("use_passed_output_shapes", 0)),
            adjust_quantized_offset_(this->template GetSingleArgument<int>( "adjust_quantized_offset", 128)) 

        lib_ = OnnxinitOnnxifiLibrary();
        backend_graph_map_ptr_ = OnnxgetOnnxBackendGraphMap();
        CAFFE_ENFORCE(lib_, "Cannot initialize ONNXIFI library");
        auto onnx_model_str =
            this->template GetSingleArgument<std::string>("onnx_model", "");
        CAFFE_ENFORCE(!onnx_model_str.empty(), "onnx_model cannot be empty");
        if (use_glow_aot_) {
          auto netdef_str =
              this->template GetSingleArgument<std::string>("netdef_str", "");
          CAFFE_ENFORCE(ParseProtoFromLargeString(netdef_str, &netdef_));
        } else if (!use_onnx_) {
          CAFFE_ENFORCE(ParseProtoFromLargeString(onnx_model_str, &netdef_));
        }

        // Setup input/output descriptor templates
        input_names_ =
            this->template GetRepeatedArgument<std::string>("input_names");
        output_names_ =
            this->template GetRepeatedArgument<std::string>("output_names");
        CAFFE_ENFORCE_EQ(input_names_.size(), operator_def.input_size());
        CAFFE_ENFORCE_EQ(output_names_.size(), operator_def.output_size());
        for (const auto& input : input_names_) {
          input_desc_.push_back(onnxTensorDescriptorV1());
          input_desc_.back().name = input.c_str();
        }
        all_offsets_.reserve(ws->Blobs().size());
        all_scales_.reserve(ws->Blobs().size());
        input_shapes_.resize(input_names_.size());
        output_shapes_max_bs_.resize(output_names_.size());
        quantized_outputs_.resize(output_names_.size(), false);
        int output_idx = 0;
        ArgumentHelper helper(operator_def);
        auto output_shape_info =
            helper.GetRepeatedArgument<TensorProto>("output_shape_info");
        auto output_qshape_info =
            helper.GetRepeatedArgument<QTensorProto>("output_qshape_info");
        std::unordered_map<std::string, TensorProto> output_shape_map;
        for (const auto& info : output_shape_info) {
          output_shape_map.emplace(info.name(), info);
        }
        std::unordered_map<std::string, QTensorProto> output_qshape_map;
        for (const auto& info : output_qshape_info) {
          output_qshape_map.emplace(info.name(), info);
        }
        bool has_quantized_output = false;
        for (const auto& output : output_names_) {
          output_desc_.push_back(onnxTensorDescriptorV1());
          output_desc_.back().name = output.c_str();

          // For output, we try to get its output size hint
          const auto it = output_shape_map.find(output);
          if (it != output_shape_map.end()) {
            output_shape_hints_.emplace(
                output_idx, details::TensorInfo(it->second));
          } else {
            const auto qit = output_qshape_map.find(output);
            if (qit != output_qshape_map.end()) {
              output_shape_hints_.emplace(
                  output_idx, details::TensorInfo(qit->second));
              quantized_outputs_[output_idx] = true;
              has_quantized_output = true;
            }
          }
          ++output_idx;
        }
        if (!has_quantized_output) {
          adjust_quantized_offset_ = 0;
        }

        LOG(INFO) << "use_onnx_=" << use_onnx_
            << ", use_glow_aot_=" << use_glow_aot_
            << ", use_passed_output_shapes_=" << use_passed_output_shapes_;

        if (use_passed_output_shapes_) {
          // Populate output_shapes_per_bs_
          for (int bs = 1; bs < max_batch_size_; ++bs) {
            auto output_shapes_tp = helper.GetRepeatedArgument<TensorProto>("output_shapes_bs_" + caffe2::to_string(bs));
            auto output_qshapes_tp = helper.GetRepeatedArgument<TensorProto>("output_qshapes_bs_" + caffe2::to_string(bs));
            CAFFE_ENFORCE_EQ(output_names_.size(), output_shapes_tp.size() + output_qshapes_tp.size());

            std::unordered_map<std::string, details::TensorInfo> name_to_shape;
            for (const auto& output_shape_tp : output_shapes_tp) {
              name_to_shape.emplace(output_shape_tp.name(), details::TensorInfo{output_shape_tp});
            }
            for (const auto& output_qshape_tp : output_qshapes_tp) {
              name_to_shape.emplace(output_qshape_tp.name(), details::TensorInfo{output_qshape_tp});
            }

            for (output_idx = 0; output_idx < output_names_.size(); ++output_idx) {
              auto it = name_to_shape.find(output_names_[output_idx]);
              CAFFE_ENFORCE(it != name_to_shape.end());
              output_shapes_per_bs_[bs].push_back({});
              auto &output_shapes = output_shapes_per_bs_[bs].back();
              std::copy(it->second.dims.cbegin(), it->second.dims.cend(), std::back_inserter(output_shapes));
            }
          }
        }

        // Get output resizing hints
        adjust_output_batch_ =
            this->template GetSingleArgument<int>("adjust_output_batch", 0);

        // Encode arguments starting with "custom_" to backend
        std::vector<uint64_t> property_pointers;
        std::vector<int64_t> int_args;
        std::vector<float> float_args;
        buildPropertyList(operator_def, &property_pointers, &int_args, &float_args);

        // Initialize the backend if it has not been already created. When we
        // initialized the backend, we will get the weights (initializers) from the
        // workspace and offload onto the backend. This should be done only once.
        // Subsequent call of this function with the same model id should find a
        // cached backend and therefore there is no need to repeat the above
        // process.
        buildBackendAndGraph(ws, property_pointers, onnx_model_str);
        */
    }
    
    #[inline] pub fn set_enable_tracing(&mut self, b: bool)  {
        
        todo!();
        /*
            enable_tracing_ = b;
        */
    }
    
    #[cfg(onnxifi_enable_ext)]
    #[inline] pub fn traces(&self) -> Arc<OnnxTraceEventList> {
        
        todo!();
        /*
            return traces_;
        */
    }
    
    #[inline] pub fn build_property_list(&mut self, 
        unused0:        &OperatorDef,
        property_list: *mut Vec<u64>,
        unused1:        *mut Vec<i64>,
        unused2:        *mut Vec<f32>)  {

        todo!();
        /*
            property_list->push_back(ONNXIFI_BACKEND_PROPERTY_NONE);
        */
    }
    
    #[inline] pub fn build_backend_and_graph(&mut self, 
        ws:                *mut Workspace,
        property_pointers: &Vec<u64>,
        onnx_model_str:    &String)  {
        
        todo!();
        /*
            op_id_string_ =
            this->template GetSingleArgument<std::string>("model_id", "") + ":" +
            this->template GetSingleArgument<std::string>("net_pos", "");

        auto initializers =
            this->template GetRepeatedArgument<std::string>("initializers");
        // Build the Onnxifi engine
        auto backend_index =
            this->template GetSingleArgument<int>("backend_id", use_onnx_ ? 1 : 0);
        // If using Glow AOT, override the backend_id to 1, since it uses a custom
        // ONNX format, and that's the id we use for the ONNX backend.
        if (use_glow_aot_) {
          backend_index = 1;
        }
        auto creator = [this,
                        ws,
                        property_pointers,
                        backend_index,
                        &onnx_model_str,
                        &initializers]() {
          std::vector<onnxBackendID> backend_ids;
          size_t num_backends{0};
          CAFFE_ENFORCE_EQ(
              lib_->onnxGetBackendIDs(nullptr, &num_backends),
              ONNXIFI_STATUS_FALLBACK);
          CAFFE_ENFORCE_GT(
              num_backends, 0, "At least 1 onnxifi backend should be available");
          CAFFE_ENFORCE_LT(
              backend_index,
              num_backends,
              "Backend idx out of bound: ",
              backend_index,
              ", #backends: ",
              num_backends);
          backend_ids.resize(num_backends);
          CAFFE_ENFORCE_EQ(
              lib_->onnxGetBackendIDs(backend_ids.data(), &num_backends),
              ONNXIFI_STATUS_SUCCESS);

          onnxBackendID backend_id = backend_ids[backend_index];
          onnxBackend backend{nullptr};

          CAFFE_ENFORCE_EQ(
              lib_->onnxInitBackend(backend_id, property_pointers.data(), &backend),
              ONNXIFI_STATUS_SUCCESS);

          // Release unused backend ids.
          for (size_t i = 0; i < num_backends; ++i) {
            if (i == backend_index) {
              continue;
            }
            lib_->onnxReleaseBackendID(backend_ids[i]);
          }

          // Get weights
          std::vector<std::string> weight_names;
          std::vector<std::vector<uint64_t>> weight_shapes;
          auto weight_descs = buildInitializationList(
              ws,
              initializers,
              &weight_names,
              &weight_shapes,
              &all_scales_,
              &all_offsets_);

          // Extra weight shapes
          std::unordered_map<std::string, ShapeInfo> weight_shape_info;
          for (size_t i = 0; i < weight_names.size(); ++i) {
            TensorShape shape;
            const auto& shape0 = weight_shapes[i];
            for (const auto d : shape0) {
              shape.add_dims(d);
            }
            weight_shape_info[weight_names[i]] = ShapeInfo(
                std::vector<TensorBoundShape::DimType>(
                    shape0.size(), TensorBoundShape_DimType_CONSTANT),
                std::move(shape));
          }

          Blob* defered_blob_reader = nullptr;
          if (ws->HasBlob("__DEFERRED_BLOB_READER__")) {
            defered_blob_reader = ws->GetBlob("__DEFERRED_BLOB_READER__");
          }
          onnxGraph graph{nullptr};

          static const uint64_t auxPropertiesListAOT[] = {
              ONNXIFI_OPTIMIZATION_AOT, ONNXIFI_GRAPH_PROPERTY_NONE};
          auto ret = lib_->onnxInitGraph(
              backend,
              use_glow_aot_ ? auxPropertiesListAOT : nullptr,
              onnx_model_str.size(),
              (const void*)(onnx_model_str.c_str()),
              weight_descs.size(),
              weight_descs.data(),
              &graph,
              static_cast<uint32_t>(max_seq_size_),
              defered_blob_reader);
          if (ret != ONNXIFI_STATUS_SUCCESS) {
            if (ret == ONNXIFI_STATUS_FATAL_ERROR) {
              C10_THROW_ERROR(
                  OnnxfiBackendSystemError, "Fatal error during onnxInitGraph");
            } else {
              CAFFE_THROW("onnxInitGraph failed");
            }
          }

          return std::make_shared<OnnxBackendGraphInfo>(
              backend_id, backend, graph, lib_, std::move(weight_shape_info));
        };
        backend_graph_shared_ptr_ =
            backend_graph_map_ptr_->insert(op_id_string_, creator);

        backend_id_ = backend_graph_shared_ptr_->backend_id;
        backend_ = backend_graph_shared_ptr_->backend;
        graph_ = backend_graph_shared_ptr_->graph;
        input_shape_info_ = backend_graph_shared_ptr_->weight_shape_info;

        getExtFunctionPointers();
        */
    }
    
    /// Set up function pointer if onnxifi_ext is enabled
    #[inline] pub fn get_ext_function_pointers(&mut self)  {
        
        todo!();
        /*
            #ifdef ONNXIFI_ENABLE_EXT
        union {
          onnxExtensionFunctionPointer p;
          decltype(onnxSetIOAndRunGraphPointer_) set;
          decltype(onnxReleaseTraceEventsPointer_) release;
          decltype(onnxWaitEventForPointer_) waitfor;
        } u;
        if (lib_->onnxGetExtensionFunctionAddress(
                backend_id_, "onnxSetIOAndRunGraphFunction", &u.p) !=
            ONNXIFI_STATUS_SUCCESS) {
          onnxSetIOAndRunGraphPointer_ = nullptr;
        } else {
          onnxSetIOAndRunGraphPointer_ = u.set;
        }
        if (lib_->onnxGetExtensionFunctionAddress(
                backend_id_, "onnxReleaseTraceEventsFunction", &u.p) !=
            ONNXIFI_STATUS_SUCCESS) {
          onnxReleaseTraceEventsPointer_ = nullptr;
        } else {
          onnxReleaseTraceEventsPointer_ = u.release;
        }
        if (lib_->onnxGetExtensionFunctionAddress(
                backend_id_, "onnxWaitEventForFunction", &u.p) !=
            ONNXIFI_STATUS_SUCCESS) {
          onnxWaitEventForPointer_ = nullptr;
        } else {
          onnxWaitEventForPointer_ = u.waitfor;
        }
    #endif
        */
    }
}

impl OnnxifiOp<CPUContext> {

    #[inline] pub fn build_initialization_list(&self, 
        ws:            *mut Workspace,
        initializers:  &Vec<String>,
        weight_names:  *mut Vec<String>,
        weight_shapes: *mut Vec<Vec<u64>>,
        all_scales:    *mut Vec<Vec<f32>>,
        all_offsets:   *mut Vec<Vec<i32>>) -> Vec<OnnxTensorDescriptorV1> {

        todo!();
        /*
            std::unordered_set<std::string> initialization_list(
          initializers.begin(), initializers.end());
      const std::vector<string>& ws_blobs = ws->Blobs();
      // Since onnxTensorDescriptorV1.name will point into the memory in
      // weight_names, we need to prevent weight_names from reallocating by
      // reserving enough memory ahead of time
      weight_names->reserve(ws_blobs.size());
      std::vector<onnxTensorDescriptorV1> descs;
      for (const auto& s : ws_blobs) {
        auto it = initialization_list.find(s);
        if (it != initialization_list.end()) {
          weight_names->emplace_back(s);
          onnxTensorDescriptorV1 tensor_desc;
          tensor_desc.name = weight_names->back().c_str();
          BlobToTensorDescriptor(
              s, ws, &tensor_desc, weight_shapes, all_scales, all_offsets);
          descs.push_back(tensor_desc);
          initialization_list.erase(it);
        }
      }
      CAFFE_ENFORCE(initialization_list.empty(), "Unfulfilled initialization list");
      return descs;
        */
    }
    
    /// initialize an OutputReshapeInfo object
    #[inline] pub fn init_output_reshape_info(&self) -> OutputReshapeInfo {
        
        todo!();
        /*
            details::OutputReshapeInfo output_reshape_info;
      output_reshape_info.begins.reserve(output_names_.size());
      output_reshape_info.ends.reserve(output_names_.size());
      output_reshape_info.fast_path.reserve(output_names_.size());
      for (int i = 0; i < output_names_.size(); ++i) {
        const auto it = output_shape_hints_.find(i);
        CAFFE_ENFORCE(
            it != output_shape_hints_.end(),
            "Cannot find output shape hints for ",
            output_names_[i]);
        int64_t num_dims = it->second.dims.size();
        // Initialize the tensors used to slice the output
        output_reshape_info.begins.emplace_back();
        ReinitializeTensor(
            &output_reshape_info.begins.back(),
            {num_dims},
            at::dtype<int32_t>().device(CPU));
        output_reshape_info.ends.emplace_back();
        ReinitializeTensor(
            &output_reshape_info.ends.back(),
            {num_dims},
            at::dtype<int32_t>().device(CPU));
      }
      return output_reshape_info;
        */
    }

    /// Helper method for extractOutputBatchSizes(), 
    /// used to deduplicate code of populating output reshape infos
    pub fn fill_output_reshape_info<DimContainer>(&mut self,
        real_shape:          &DimContainer,
        max_shape:           &[u64],
        output_reshape_info: &mut OutputReshapeInfo,
        current_index:       i32) 
    {
        todo!();

        /*
          CAFFE_ENFORCE_EQ(real_shape.size(), max_shape.size());
          const auto dim_size = real_shape.size();
          auto& begin = output_reshape_info.begins[currentIndex];
          begin.Resize(dim_size);
          int32_t* begin_ptr = begin.template mutable_data<int32_t>();
          auto& end = output_reshape_info.ends[currentIndex];
          end.Resize(dim_size);
          int32_t* end_ptr = end.template mutable_data<int32_t>();
          int32_t mismatch = 0;
          for (int j = 0; j < dim_size; ++j) {
            CAFFE_ENFORCE_GE(
                max_shape[j],
                real_shape[j],
                "It is weird that max shape of ",
                output_names_[currentIndex],
                " is smaller than real shape at dim ",
                j,
                " (",
                max_shape[j],
                " vs ",
                real_shape[j],
                ")");
            begin_ptr[j] = 0;
            if (max_shape[j] > real_shape[j]) {
              end_ptr[j] = real_shape[j];
              mismatch += j;
            } else {
              end_ptr[j] = max_shape[j];
            }
          }

          if (dim_size > 0) {
            output_reshape_info.fast_path[currentIndex] = !mismatch;
          } else {
            output_reshape_info.fast_path[currentIndex] = false;
          }
        */
    }

    /**
      | Extract output batch size. If the output
      | batch size is going to be at
      | max_batch_size_, return true indicating
      | that no output shape adjustment is
      | needed. Otherwise, return false.
      */
    #[inline] pub fn extract_output_batch_sizes(&mut self) -> i32 {
        
        todo!();
        /*
            if (use_onnx_ || !adjust_output_batch_) {
        return max_batch_size_;
      }

      // Get the real batch size from nominal input. If it's equal to
      // max_batch_size, mark that we don't need to adjust batch size and return.
      // Otherwise, do a pass of shape inference to get the real shapes of the
      // outputs.
      const Tensor* t = nullptr;
      if (this->template InputIsType<int8::Int8TensorCPU>(nominal_batch_idx_)) {
        const auto& input_tensor_int8 =
            this->template Input<int8::Int8TensorCPU>(nominal_batch_idx_);
        t = &input_tensor_int8.t;
      } else {
        t = &Input(nominal_batch_idx_);
      }

      CAFFE_ENFORCE(
          t, "Null input shape tensor ptr. Possibly unsupported tensor type");
      CAFFE_ENFORCE(
          !t->sizes().empty(),
          input_names_[nominal_batch_idx_],
          " cannot be empty");
      const auto dims = t->sizes();
      const int current_batch_size = dims[0];
      if (current_batch_size == max_batch_size_) {
        return max_batch_size_;
      }

      // We still need to adjust output size but we can skip the shape inference as
      // it was done before.
      if (output_reshape_info_.count(current_batch_size)) {
        return current_batch_size;
      }

      auto& output_reshape_info =
          output_reshape_info_.emplace(current_batch_size, initOutputReshapeInfo())
              .first->second;

      if (use_passed_output_shapes_) {
        auto shape_info_it = output_shapes_per_bs_.find(current_batch_size);
        CAFFE_ENFORCE(
            shape_info_it != output_shapes_per_bs_.end(),
            "Unable to find outputs shapes for bs=",
            current_batch_size);
        CAFFE_ENFORCE_EQ(shape_info_it->second.size(), OutputSize());

        for (int i = 0; i < OutputSize(); ++i) {
          fillOutputReshapeInfo(
              shape_info_it->second[i],
              output_shapes_max_bs_[i],
              output_reshape_info,
              i);
        }
      } else {
        BoundShapeSpec spec(dims[0], max_seq_size_);
        auto bound_shape_inferencer =
            BoundShapeInferencerRegistry()->Create("C10", spec);
        for (int i = 0; i < InputSize(); ++i) {
          at::IntArrayRef dim0;
          bool quantized = false;
          if (this->template InputIsType<int8::Int8TensorCPU>(i)) {
            const auto& input_tensor_int8 =
                this->template Input<int8::Int8TensorCPU>(i);
            const auto& t0 = input_tensor_int8.t;
            dim0 = t0.sizes();
            quantized = true;
          } else {
            const auto& t0 = Input(i);
            dim0 = t0.sizes();
          }
          TensorShape shape;
          for (const auto d : dim0) {
            shape.add_dims(d);
          }
          std::vector<TensorBoundShape::DimType> dim_type(
              shape.dims_size(), TensorBoundShape_DimType_CONSTANT);
          if (dim_type.size()) {
            dim_type[0] = TensorBoundShape_DimType_BATCH;
          }
          input_shape_info_[input_names_[i]] =
              ShapeInfo(dim_type, std::move(shape), quantized);
        }
        bound_shape_inferencer->InferBoundShapeAndType(
            netdef_, input_shape_info_, nullptr, false);
        const auto& shape_info = bound_shape_inferencer->shape_info();
        for (int i = 0; i < OutputSize(); ++i) {
          const auto find_res = shape_info.find(output_names_[i]);
          CAFFE_ENFORCE(find_res != shape_info.end());
          fillOutputReshapeInfo(
              find_res->second.shape.dims(),
              output_shapes_max_bs_[i],
              output_reshape_info,
              i);
        }
      }

      return current_batch_size;
        */
    }

    /**
      | Adjust output tensor shape based on the
      | current input batch size.
      |
      | If the output shape is conditioned on
      | first dim (batch size), we have a fast
      | path to shrink the tensor shape by just
      | manipulating the meta data.
      |
      | Otherwise, we have to slice it in the
      | middle of the dimension with copy
      | invoked. This is a slow path and we don't
      | expect it to happen very often.
      |
      | We can already omit this step by setting
      | "adjust_output_batch_" to false
      */
    #[inline] pub fn adjust_output_batch_sizes(&mut self, current_batch_size: i32)  {
        
        todo!();
        /*
            auto it = output_reshape_info_.find(current_batch_size);
      CAFFE_ENFORCE(
          it != output_reshape_info_.end(),
          "Cannot find current_batch_size ",
          current_batch_size,
          " in output_reshape_info_");
      const auto& output_reshape_info = it->second;
      CPUContext context;
      Tensor tmp(CPU);
      for (int i = 0; i < OutputSize(); ++i) {
        Tensor* output_tensor = quantized_outputs_[i]
            ? (&this->template Output<int8::Int8TensorCPU>(i)->t)
            : Output(i);
        const auto& end = output_reshape_info.ends[i];
        if (output_reshape_info.fast_path[i]) {
          output_tensor->ShrinkTo(end.data<int32_t>()[0]);
        } else {
          // We need to use generic Slice
          SliceImpl<int32_t, CPUContext>(
              &tmp, *output_tensor, output_reshape_info.begins[i], end, &context);
          output_tensor->CopyFrom(tmp);
        }
      }
        */
    }

    /**
      | Second argument is a cache vector to
      | avoid repeated reallocation.
      | 
      | The existence of this is not ideal, which
      | is purely due to the fact that we use int64_t
      | for c2::tensor dim but uint64_t for
      | onnxDesciptor dim.
      | 
      | Maybe we should just use int64_t.
      |
      */
    pub fn set_output_shape_and_type(
        &mut self, 
        output_idx: i32, 
        tensor_dims_int64: SmallVec<[i64; 4]>) 
    {
        todo!();

        /*
          tensor_dims_int64.clear();
          std::vector<size_t> tensor_dims;
          uint64_t type = ONNXIFI_DATATYPE_FLOAT32;
          const auto it = output_shape_hints_.find(output_idx);
          CAFFE_ENFORCE(
              it != output_shape_hints_.end(),
              "Cannot find shape hint for output: ",
              output_names_[output_idx]);
          const auto& info = it->second;
          std::copy(
              info.dims.begin(), info.dims.end(), std::back_inserter(tensor_dims));
          type = it->second.onnxifi_type;
          auto& tensor_descriptor = output_desc_[output_idx];
          tensor_descriptor.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
          tensor_descriptor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
          tensor_descriptor.dimensions = tensor_dims.size();
          CAFFE_ENFORCE(
              tensor_descriptor.dimensions != 0, tensor_descriptor.name, " has 0 dim");
          auto& output_shape = output_shapes_max_bs_[output_idx];
          output_shape.clear();
          output_shape.insert(
              output_shape.begin(), tensor_dims.cbegin(), tensor_dims.cend());
          tensor_descriptor.shape = output_shape.data();
          std::copy(
              tensor_dims.cbegin(),
              tensor_dims.cend(),
              std::back_inserter(tensor_dims_int64));

          // Setup the output C2 tensor
          if (!info.quantized) {
            // Normal Tensor
            auto* output_tensor = Output(
                output_idx,
                tensor_dims_int64,
                at::dtype(OnnxifiTypeToDataType(type)).device(CPU));
            setOutputTensorDescriptorTypeAndBuffer(
                type, output_tensor, &tensor_descriptor);
          } else if (info.quantizationParams == 1) {
            // single quantizer, output Int8Tensor
            auto* output_tensor =
                this->template Output<int8::Int8TensorCPU>(output_idx);
            output_tensor->t.Resize(tensor_dims_int64);
            setOutputTensorDescriptorTypeAndBuffer(
                type, &output_tensor->t, &tensor_descriptor);
            tensor_descriptor.quantizationParams = 1;
            tensor_descriptor.quantizationAxis = 1;
            tensor_descriptor.scales = &output_tensor->scale;
            tensor_descriptor.biases = &output_tensor->zero_point;
          } else {
            CAFFE_THROW(
                "OnnxifiOp does not support output tensor with multi-quantization params: ",
                output_names_[output_idx]);
          }

        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(input_desc_.size(), InputSize());
      for (unsigned i = 0U; i < InputSize(); ++i) {
        auto& tensor_descriptor = input_desc_[i];
        tensor_descriptor.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
        tensor_descriptor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
        at::IntArrayRef tensor_dims;
        if (this->template InputIsType<int8::Int8TensorCPU>(i)) {
          const auto& input_tensor_int8 =
              this->template Input<int8::Int8TensorCPU>(i);
          const auto& cpu_tensor = input_tensor_int8.t;
          tensor_dims = cpu_tensor.sizes();
          setInputTensorDescriptorTypeAndBuffer(
              input_tensor_int8, &tensor_descriptor);
        } else {
          const auto& input_tensor = Input(i);
          tensor_dims = input_tensor.sizes();
          setInputTensorDescriptorTypeAndBuffer(input_tensor, &tensor_descriptor);
        }
        auto& input_shape = input_shapes_[i];
        input_shape.clear();
        input_shape.insert(
            input_shape.begin(), tensor_dims.cbegin(), tensor_dims.cend());
        tensor_descriptor.dimensions = tensor_dims.size();
        tensor_descriptor.shape = input_shape.data();
      }

      CAFFE_ENFORCE_EQ(output_desc_.size(), OutputSize());
      c10::SmallVector<int64_t, 4> tensor_dims_int64;
      for (unsigned i = 0U; i < OutputSize(); ++i) {
        setOutputShapeAndType(i, tensor_dims_int64);
      }
      bool ext_supported = false;
      onnxMemoryFenceV1 input_fence;
      onnxMemoryFenceV1 output_fence;
      std::vector<int> output_batch_sizes;
      int current_batch_size = max_batch_size_;
    #ifdef ONNXIFI_ENABLE_EXT
      /**
       * If onnxifi extension mode is enabled,
       * and onnxSetIOAndRunGraph is supported in backend,
       * then we run throw this workflow;
       * Else we fallback to non-onnxifi-extension workflow.
       **/
      if (onnxSetIOAndRunGraphPointer_ != nullptr) {
        ext_supported = true;
        output_fence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
        output_fence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
        traces_.reset();
        if (enable_tracing_) {
          traces_ = std::shared_ptr<onnxTraceEventList>(
              new onnxTraceEventList(), [this](onnxTraceEventList* p) {
                if (p && onnxReleaseTraceEventsPointer_) {
                  CAFFE_ENFORCE_EQ(
                      (*onnxReleaseTraceEventsPointer_)(p), ONNXIFI_STATUS_SUCCESS);
                }
                delete p;
              });
          traces_->numEvents = 0;
        }

        const onnxStatus status = (*onnxSetIOAndRunGraphPointer_)(
            graph_,
            input_desc_.size(),
            input_desc_.data(),
            output_desc_.size(),
            output_desc_.data(),
            &output_fence,
            traces_.get());
        CAFFE_ENFORCE_EQ(
            status,
            ONNXIFI_STATUS_SUCCESS,
            "Reason: onnxSetIOAndRunGraph returned status code ",
            mapOnnxStatusToString(status));

        current_batch_size = extractOutputBatchSizes();
        onnxEventState eventState;
        onnxStatus eventStatus;
        std::string message;
        size_t messageLength = 512;
        message.resize(messageLength);

        CAFFE_ENFORCE_EQ(
            (*onnxWaitEventForPointer_)(
                output_fence.event,
                timeout_,
                &eventState,
                &eventStatus,
                const_cast<char*>(message.data()),
                &messageLength),
            ONNXIFI_STATUS_SUCCESS);
        CAFFE_ENFORCE_EQ(
            eventState,
            ONNXIFI_EVENT_STATE_SIGNALLED,
            "Onnxifi run timeouted out after ",
            timeout_,
            " ms.");
        if (eventStatus != ONNXIFI_STATUS_SUCCESS) {
          if (messageLength == 0) {
            CAFFE_THROW("onnxifi internal error");
          } else {
            CAFFE_THROW(message);
          }
        }
        CAFFE_ENFORCE_EQ(
            lib_->onnxReleaseEvent(output_fence.event), ONNXIFI_STATUS_SUCCESS);
      }
    #endif
      if (!ext_supported) {
        CAFFE_ENFORCE_EQ(
            lib_->onnxSetGraphIO(
                graph_,
                input_desc_.size(),
                input_desc_.data(),
                output_desc_.size(),
                output_desc_.data()),
            ONNXIFI_STATUS_SUCCESS);

        input_fence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
        input_fence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
        CAFFE_ENFORCE_EQ(
            lib_->onnxInitEvent(backend_, &input_fence.event),
            ONNXIFI_STATUS_SUCCESS);
        output_fence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
        output_fence.type = ONNXIFI_SYNCHRONIZATION_EVENT;

        // Call the async run on backend, signal event on input fence and wait for
        // the event on output fence
        CAFFE_ENFORCE_EQ(
            lib_->onnxRunGraph(graph_, &input_fence, &output_fence),
            ONNXIFI_STATUS_SUCCESS);
        CAFFE_ENFORCE_EQ(
            lib_->onnxSignalEvent(input_fence.event), ONNXIFI_STATUS_SUCCESS);
        current_batch_size = extractOutputBatchSizes();
        CAFFE_ENFORCE_EQ(
            lib_->onnxWaitEvent(output_fence.event), ONNXIFI_STATUS_SUCCESS);

        // Destroy the event objects
        CAFFE_ENFORCE_EQ(
            lib_->onnxReleaseEvent(input_fence.event), ONNXIFI_STATUS_SUCCESS);
        CAFFE_ENFORCE_EQ(
            lib_->onnxReleaseEvent(output_fence.event), ONNXIFI_STATUS_SUCCESS);
      }

      if (adjust_quantized_offset_) {
        for (unsigned i = 0U; i < OutputSize(); ++i) {
          if (quantized_outputs_[i]) {
            auto* int8_tensor = this->template Output<int8::Int8TensorCPU>(i);
            int8_tensor->zero_point += adjust_quantized_offset_;
            adjustQuantizedOffset(&int8_tensor->t, adjust_quantized_offset_);
          }
        }
      }

      if (adjust_output_batch_ && current_batch_size != max_batch_size_) {
        adjustOutputBatchSizes(current_batch_size);
      }
      enable_tracing_ = false;
      return true;
        */
    }
}

