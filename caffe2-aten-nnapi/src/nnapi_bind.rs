crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/nnapi/nnapi_bind.cpp]

lazy_static!{
    /*
    nnapi_wrapper* nnapi;
    nnapi_wrapper* check_nnapi;
    */
}

pub fn load_platform_library()  {
    
    todo!();
        /*
            static int run_once = [](){
        nnapi_wrapper_load(&nnapi, &check_nnapi);
        CAFFE_ENFORCE(nnapi);
        CAFFE_ENFORCE(nnapi->Model_free);
        CAFFE_ENFORCE(nnapi->Compilation_free);
        CAFFE_ENFORCE(nnapi->Execution_free);
        return 0;
      }();
      (void)run_once;
        */
}

#[macro_export] macro_rules! make_smart_ptr {
    ($type:ident) => {
        /*
        
          struct type ## Freer { 
            void operator()(ANeuralNetworks ## type * obj) { 
              if (!nnapi) { /* obj must be null. */ return; } 
              nnapi-> type ## _free(obj); 
            } 
          }; 
          typedef unique_ptr<ANeuralNetworks ## type, type ## Freer> type ## Ptr;
        */
    }
}

make_smart_ptr!{Model}
make_smart_ptr!{Compilation}
make_smart_ptr!{Execution}

pub struct NnapiCompilation {
    base:        TorchJitCustomClassHolder,
    model:       ModelPtr,
    compilation: CompilationPtr,
    num_inputs:  i32,
    num_outputs: i32,
}

impl Default for NnapiCompilation {
    
    fn default() -> Self {
        todo!();
        /*


            // Could possibly call load_platform_library here, but error reporting
        // can be complicated if the constructor is called during model loading.
        // Instead, delay all work until the explicit init call.
        */
    }
}

impl NnapiCompilation {
    
    pub fn init(&mut self, 
        serialized_model_tensor: Tensor,
        parameter_buffers:       Vec<Tensor>)  {
        
        todo!();
        /*
            TORCH_CHECK(!model_, "Attempted to re-initialize NnapiCompilation.");

        load_platform_library();

        vector<const void*> buffers;
        vector<i32> buffer_sizes;
        for (auto& t : parameter_buffers) {
          TORCH_CHECK(t.is_contiguous());
          buffers.push_back(t.data_ptr());
          buffer_sizes.push_back(t.nbytes());
        }

        TORCH_CHECK(serialized_model_tensor.is_contiguous());
        // This is currently always i32, but support u8 for old models
        // and possible future changes to the generator.
        u8* ser_model_ptr =
          serialized_model_tensor.scalar_type() == ScalarType::Byte
            ? serialized_model_tensor.data_ptr<u8>()
            : reinterpret_cast<u8*>(serialized_model_tensor.data_ptr<i32>());
        ArrayRef<u8> ser_model = {
          ser_model_ptr,
          serialized_model_tensor.nbytes()
        };
        TORCH_CHECK(ser_model.size() > 0);

        ANeuralNetworksModel* model;
        check_nnapi->Model_create(&model);
        CAFFE_ENFORCE(model);
        model_.reset(model);

        int load_result = ::nnapi::load_nnapi_model(
            nnapi,
            model_.get(),
            ser_model.data(),
            ser_model.size(),
            buffers.size(),
            buffers.data(),
            buffer_sizes.data(),
            0,
            nullptr,
            nullptr,
            &num_inputs_,
            &num_outputs_,
            nullptr);
        CAFFE_ENFORCE(load_result == 0);

        check_nnapi->Model_finish(model_.get());

        ANeuralNetworksCompilation* compilation;
        check_nnapi->Compilation_create(model_.get(), &compilation);
        // TODO: Make this configurable.
        check_nnapi->Compilation_setPreference(compilation, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
        check_nnapi->Compilation_finish(compilation);
        compilation_.reset(compilation);
        */
    }
    
    pub fn run(&mut self, 
        inputs:  Vec<Tensor>,
        outputs: Vec<Tensor>)  {
        
        todo!();
        /*
        ANeuralNetworksExecution* execution;
        check_nnapi->Execution_create(compilation_.get(), &execution);
        ExecutionPtr execution_unique_ptr(execution);

        TORCH_CHECK((i32)inputs.size() == num_inputs_);
        TORCH_CHECK((i32)outputs.size() == num_outputs_);

        for (usize i = 0; i < inputs.size(); i++) {
          auto& t = inputs[i];
          // TODO: Check contiguous and dtype.
          ANeuralNetworksOperandType op_type;
          vector<u32> dim;
          get_operand_type(t, &op_type, &dim);
          check_nnapi->Execution_setInput(
              execution,
              i,
              &op_type,
              t.data_ptr(),
              t.nbytes());
        }

        for (usize i = 0; i < outputs.size(); i++) {
          auto& t = outputs[i];
          // TODO: Check contiguous and dtype.
          check_nnapi->Execution_setOutput(
              execution,
              i,
              nullptr,
              t.data_ptr(),
              t.nbytes());
        }

        check_nnapi->Execution_compute(execution);

        // TODO: Maybe skip this for fixed-size outputs?
        for (usize i = 0; i < outputs.size(); i++) {
          auto& t = outputs[i];
          u32 rank;
          check_nnapi->Execution_getOutputOperandRank(execution, i, &rank);
          vector<u32> dims(rank);
          check_nnapi->Execution_getOutputOperandDimensions(execution, i, dims.data());
          vector<i64> long_dims(dims.begin(), dims.end());
          // TODO: Maybe check that only the batch dimension is changed?
          t.resize_(long_dims);
        }
        */
    }
    
    pub fn get_operand_type(
        t:       &Tensor,
        operand: *mut ANeuralNetworksOperandType,
        dims:    *mut Vec<u32>)  {
        
        todo!();
        /*
            operand->dimensionCount = t.dim();
        TORCH_CHECK(operand->dimensionCount == t.dim()); // Check for overflow.
        dims->resize(t.dim());
        operand->dimensions = dims->data();
        for (usize i = 0; i < dims->size(); i++) {
          (*dims)[i] = t.sizes()[i];
          TORCH_CHECK((*dims)[i] == t.sizes()[i]); // Check for overflow.
        }
        if (t.scalar_type() == kFloat) {
          operand->type = ANEURALNETWORKS_TENSOR_FLOAT32;
          operand->scale = 0;
          operand->zeroPoint = 0;
          return;
        }
        if (t.scalar_type() == kQUInt8) {
          TORCH_CHECK(t.is_quantized());
          operand->type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
          operand->scale = t.q_scale();
          operand->zeroPoint = t.q_zero_point();
          return;
        }
        // TODO: Support more dtypes.
        CAFFE_THROW("Bad dtype");
        */
    }
}

#[cfg(not(__APPLE__))]
lazy_static!{
    /*
    static auto register_NnapiCompilation = [](){
      try {
        return TorchJitclass_<NnapiCompilation>("_nnapi", "Compilation")
            .def(TorchJitinit<>())
            .def("init", &NnapiCompilation::init)
            .def("run", &NnapiCompilation::run)
            ;
      } catch (exception& exn) {
        LOG(ERROR) << "Failed to register class nnapi.Compilation: " << exn.what();
        throw;
      }
    }();
    */
}
