crate::ix!();

declare_bool!{caffe2_dnnlowp_force_scale_power_of_two}
declare_bool!{caffe2_dnnlowp_preserve_activation_sparsity}
declare_bool!{caffe2_dnnlowp_preserve_weight_sparsity}
declare_bool!{caffe2_force_shared_col_buffer}
declare_bool!{caffe2_net_async_check_stream_status}
declare_bool!{caffe2_net_async_profile_operators}
declare_bool!{caffe2_net_async_run_root_tasks_inline}
declare_bool!{caffe2_net_async_use_per_net_pools}
declare_bool!{caffe2_net_async_use_single_pool}
declare_bool!{caffe2_operator_throw_if_fp_exceptions}
declare_bool!{caffe2_operator_throw_if_fp_overflow_exceptions}
declare_bool!{caffe2_operator_throw_on_first_occurrence_if_fp_exceptions}
declare_bool!{caffe2_print_blob_sizes_at_exit}
declare_bool!{caffe2_report_cpu_memory_usage}
declare_bool!{caffe2_rnn_executor}
declare_bool!{caffe2_serialize_fp16_as_bytes}
declare_bool!{caffe2_serialize_using_bytes_as_holder}
declare_bool!{caffe2_workspace_stack_debug}

declare_double!{caffe2_dnnlowp_activation_p99_threshold}
declare_double!{caffe2_dnnlowp_weight_p99_threshold}

declare_export_caffe2_op_to_c10!{ AliasWithName }
declare_export_caffe2_op_to_c10!{ BBoxTransform }
declare_export_caffe2_op_to_c10!{ BatchBoxCox }
declare_export_caffe2_op_to_c10!{ BatchPermutation }
declare_export_caffe2_op_to_c10!{ BoxWithNMSLimit }
declare_export_caffe2_op_to_c10!{ Bucketize }
declare_export_caffe2_op_to_c10!{BatchBucketOneHot}
declare_export_caffe2_op_to_c10!{CollectAndDistributeFpnRpnProposals}
declare_export_caffe2_op_to_c10!{CollectRpnProposals}
declare_export_caffe2_op_to_c10!{CopyCPUToGPU}
declare_export_caffe2_op_to_c10!{CopyGPUToCPU}
declare_export_caffe2_op_to_c10!{DistributeFpnProposals}
declare_export_caffe2_op_to_c10!{Fused8BitRowwiseQuantizedToFloat}
declare_export_caffe2_op_to_c10!{GatherRangesOp}
declare_export_caffe2_op_to_c10!{GatherRangesToDense}
declare_export_caffe2_op_to_c10!{Gelu}
declare_export_caffe2_op_to_c10!{GenerateProposals}
declare_export_caffe2_op_to_c10!{HeatmapMaxKeypoint}
declare_export_caffe2_op_to_c10!{IndexHash}
declare_export_caffe2_op_to_c10!{LSTMOp}
declare_export_caffe2_op_to_c10!{LayerNorm}
declare_export_caffe2_op_to_c10!{LearningRate}
declare_export_caffe2_op_to_c10!{LengthsGatherOp}
declare_export_caffe2_op_to_c10!{Logit}
declare_export_caffe2_op_to_c10!{MergeIdLists}
declare_export_caffe2_op_to_c10!{PackSegments}
declare_export_caffe2_op_to_c10!{PiecewiseLinearTransform}
declare_export_caffe2_op_to_c10!{RoIAlignGradient}
declare_export_caffe2_op_to_c10!{RoIAlign}
declare_export_caffe2_op_to_c10!{UnpackSegments}

declare_int!{caffe2_max_tensor_serializer_threads}
declare_int!{caffe2_net_async_max_gpus}
declare_int!{caffe2_net_async_max_numa_nodes}
declare_int!{caffe2_net_async_thread_pool_size}
declare_int!{caffe2_streams_per_gpu}
declare_int!{caffe2_tensor_chunk_size}

declare_int32!{caffe2_dnnlowp_activation_quantization_precision}
declare_int32!{caffe2_dnnlowp_eltwise_quantization_precision}
declare_int32!{caffe2_dnnlowp_requantization_multiplier_precision}
declare_int32!{caffe2_dnnlowp_weight_quantization_precision}

declare_registry!{ /* TransformRegistry, Transform */}

declare_string!{ caffe_test_root }
declare_string!{caffe2_dnnlowp_activation_quantization_kind}
declare_string!{caffe2_dnnlowp_weight_quantization_kind }
declare_string!{caffe2_override_executor}
declare_string!{caffe_test_root}

declare_typed_registry!{ /* ExternalTensorFunctionsBaseRegistry, TypeIdentifier, ExternalTensorFunctionsBase, std::unique_ptr */ }
