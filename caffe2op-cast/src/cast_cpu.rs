crate::ix!();

impl CastOp<CPUContext> {

    #[inline] pub fn do_run_with_dst_type<DstType>(&mut self) -> bool {
        todo!();
        /*
            return DispatchHelper<
              TensorTypes<
                  float,
                  int32_t,
                  bool,
                  uint8_t,
                  int8_t,
                  uint16_t,
                  int16_t,
                  int64_t,
                  double>,
              DstType>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_cpu_with_type<DstType, SrcType>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(0);

          auto* output = Output(0, input.sizes(), at::dtype<DstType>());
          const auto* data = input.template data<SrcType>();
          auto* out = output->template mutable_data<DstType>();
          auto N = input.numel();
          for (int64_t i = 0; i < N; ++i) {
            out[i] = CastHelper<DstType, SrcType>::call(data[i]);
          }
          return true;
        */
    }
    
    /// Allow for Context-specific implementations
    #[inline] pub fn set_body(&mut self, to: TensorProto_DataType)  {
        
        todo!();
        /*
            switch (to) {
        case TensorProto_DataType_FLOAT:
          // body_ = &CastOp::DoRunIncFp16WithDstType<float>;
          body_ = &CastOp<CPUContext>::DoRunWithDstType<float>;
          break;
        case TensorProto_DataType_INT32:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<int>;
          break;
        case TensorProto_DataType_BYTE:
          LOG(FATAL) << "BYTE is deprecated";
          break;
        case TensorProto_DataType_STRING:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<std::string>;
          break;
        case TensorProto_DataType_BOOL:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<bool>;
          break;
        case TensorProto_DataType_UINT8:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<uint8_t>;
          break;
        case TensorProto_DataType_INT8:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<int8_t>;
          break;
        case TensorProto_DataType_UINT16:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<uint16_t>;
          break;
        case TensorProto_DataType_INT16:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<int16_t>;
          break;
        case TensorProto_DataType_INT64:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<int64_t>;
          break;
        case TensorProto_DataType_FLOAT16:
          CAFFE_THROW("Casting to and from at::Half on CPU is not supported yet");
          // break;
        case TensorProto_DataType_DOUBLE:
          // body_ = &CastOp::DoRunIncFp16WithDstType<double>;
          body_ = &CastOp<CPUContext>::DoRunWithDstType<double>;
          break;
        case TensorProto_DataType_UNDEFINED:
          CAFFE_THROW("Cast op must have 'to' argument of type DataType");
          // break;
        default:
          CAFFE_THROW("Unexpected 'to' argument value: ", to);
      }
        */
    }
}

register_cpu_operator!{Cast, CastOp<CPUContext>}
