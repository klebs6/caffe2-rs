crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cudnn/Conv_v8.cpp]

#[cfg(all(HAS_CUDNN_V8,AT_CUDNN_ENABLED))]
pub fn get_alignment(t: &Tensor) -> u8 {
    
    todo!();
        /*
            // alignment are in bytes
      u8 alignment = 1;
      u64 address = reinterpret_cast<u64>(t.data_ptr());
      while (address % alignment == 0 && alignment < 16) alignment *= 2;
      return alignment;
        */
}

#[cfg(all(HAS_CUDNN_V8,AT_CUDNN_ENABLED))]
pub fn get_tensor_descriptor(
        t:         &Tensor,
        id:        i64,
        alignment: u8) -> CuDnnFrontendTensor {
    
    todo!();
        /*
            auto shape = t.sizes();
      auto strides = t.strides();
      return CuDnnFrontendTensorBuilder()
        .setDim(shape.size(), shape.data())
        .setStrides(strides.size(), strides.data())
        .setId(id)
        .setAlignment(alignment)
        .setDataType(getCudnnDataType(t))
        .build();
        */
}

#[cfg(all(HAS_CUDNN_V8,AT_CUDNN_ENABLED))]
pub fn get_conv_descriptor(
        data_type: CudnnDataType,
        padding:   &[i32],
        stride:    &[i32],
        dilation:  &[i32]) -> CuDnnFrontendConvDesc_v8 {
    
    todo!();
        /*
            u64 convDim = stride.size();
      return CuDnnFrontendConvDescBuilder()
        .setDataType(dataType)
        .setMathMode(CUDNN_CROSS_CORRELATION)
        .setNDims(convDim)
        .setStrides(convDim, stride.data())
        .setPrePadding(convDim, padding.data())
        .setPostPadding(convDim, padding.data())
        .setDilation(convDim, dilation.data())
        .build();
        */
}

#[cfg(all(HAS_CUDNN_V8,AT_CUDNN_ENABLED))]
pub fn filter_engine_configs(
        from:          &mut CuDnnFrontendEngineConfigList,
        to:            &mut CuDnnFrontendEngineConfigList,
        deterministic: bool,
        allow_tf32:    bool,
        scalar_type:   ScalarType)  {
    
    todo!();
        /*
            auto filter = [=](cudnnBackendDescriptor_t c) {
        if (deterministic) {
          if (CuDnnFrontendhasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) return true;
        }
        if (scalar_type == kFloat || !allow_tf32) {
          if (CuDnnFrontendhasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) return true;
          if (CuDnnFrontendhasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) return true;
        }
        return false;
      };
      CuDnnFrontendfilter(from, to, filter);
        */
}

#[cfg(all(HAS_CUDNN_V8,AT_CUDNN_ENABLED))]
pub struct CacheKey {
    params:           ConvolutionParams,
    input_alignment:  u8,
    weight_alignment: u8,
    output_alignment: u8,
}

// FIXME: make this thread-safe by reusing the
// benchmark cache in Conv_v7.cpp
//
#[cfg(all(HAS_CUDNN_V8,AT_CUDNN_ENABLED))]
lazy_static!{
    /*
    unordered_map<CacheKey, CuDnnFrontendManagedOpaqueDescriptor, ParamsHash<CacheKey>, ParamsEqual<CacheKey>> engine_cache;
    */
}

#[cfg(all(HAS_CUDNN_V8,AT_CUDNN_ENABLED))]
pub fn raw_cudnn_convolution_forward_out(
        output:        &Tensor,
        input:         &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        allow_tf32:    bool)  {
    
    todo!();
        /*
            TORCH_CHECK(!benchmark, "not supported yet");
      if (output.numel() == 0) {
        return;
      }

      cudnnHandle_t handle = getCudnnHandle();

      CacheKey key;
      setConvolutionParams(&key.params, input, weight, padding, stride, dilation, groups, deterministic, allow_tf32);
      key.input_alignment = getAlignment(input);
      key.output_alignment = getAlignment(output);
      key.weight_alignment = getAlignment(weight);

      auto run = [&](CuDnnFrontendManagedOpaqueDescriptor cfg) {
        auto plan = CuDnnFrontendExecutionPlanBuilder()
            .setHandle(handle)
            .setEngineConfig(cfg)
            .build();

        auto workspace_size = plan.getWorkspaceSize();
        auto workspace = empty({workspace_size}, input.options().dtype(kByte));
        void *data_ptrs[] = {input.data_ptr(), output.data_ptr(), weight.data_ptr()};
        // cout << plan.describe() << " requires workspace " << workspace_size << endl;
        i64 uids[] = {'x', 'y', 'w'};
        auto variantPack = CuDnnFrontendVariantPackBuilder()
            .setWorkspacePointer(workspace.data_ptr())
            .setDataPointers(3, data_ptrs)
            .setUids(3, uids)
            .build();
        AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
      };

      auto search = engine_cache.find(key);
      if (search != engine_cache.end()) {
        run(search->second);
        return;
      }

      auto op = CuDnnFrontendOperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
          .setxDesc(getTensorDescriptor(input, 'x', key.input_alignment))
          .setyDesc(getTensorDescriptor(output, 'y', key.output_alignment))
          .setwDesc(getTensorDescriptor(weight, 'w', key.weight_alignment))
          .setcDesc(getConvDescriptor(key.params.dataType, padding, stride, dilation))
          .build();
      // cout << op.describe() << endl;

      array<CuDnnFrontendOperation const *, 1> ops = {&op};

      auto opGraph = CuDnnFrontendOperationGraphBuilder()
          .setHandle(handle)
          .setOperationGraph(1, ops.data())
          .build();
      // cout << opGraph.describe() << endl;

      auto heuristics = CuDnnFrontendEngineHeuristicsBuilder()
          .setOperationGraph(opGraph)
          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
          .build();
      auto fallback = CuDnnFrontendEngineFallbackListBuilder()
                        .setOperationGraph(opGraph)
                        .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                        .build();

      auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
      auto& fallback_list = fallback.getFallbackList();

      CuDnnFrontendEngineConfigList filtered_configs;
      filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, input.scalar_type());
      filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, input.scalar_type());

      for (auto &cfg : filtered_configs) {
        try {
          run(cfg);
          engine_cache[key] = cfg;
          return;
        } catch (CuDnnFrontendcudnnException &e) {} catch(CuDNNError &e) {}
      }
      TORCH_CHECK(false, "Unable to find an engine to execute this computation");
        */
}
