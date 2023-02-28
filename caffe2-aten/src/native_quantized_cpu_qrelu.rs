// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qrelu.cpp]

define_dispatch!{qrelu_stub}
define_dispatch!{qrelu6_stub}
define_dispatch!{qrelu_leaky_stub}

#[cfg(USE_PYTORCH_QNNPACK)]
pub fn qnnpack_relu(input: Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor qy;
      TORCH_CHECK(
          input.ndimension() > 0, "qnnpack_relu(): Got empty input tensor");

      Tensor input_contig = input.contiguous(input.suggest_memory_format());

      const auto zero_point = input_contig.q_zero_point();

      initQNNPACK();

      usize num_elems = 1;
      for (int i = 1; i < input_contig.ndimension(); ++i) {
        num_elems *= input_contig.size(i);
      }

      pytorch_qnnp_operator_t qnnpack_operator{nullptr};

      const pytorch_qnnp_status createStatus = pytorch_qnnp_create_clamp_nc_u8(
          num_elems /* channels */,
          zero_point /* output min */,
          u8::max /* output max */,
          0 /* flags */,
          &qnnpack_operator);

      unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
          qnnpack_uniq_ptr(qnnpack_operator);

      TORCH_INTERNAL_ASSERT(
          createStatus == pytorch_qnnp_status_success,
          "failed to create QNNPACK Relu operator");

      qy = _empty_affine_quantized(
          input_contig.sizes(),
          device(kCPU).dtype(input.scalar_type()),
          input_contig.q_scale(),
          input_contig.q_zero_point(),
          input.suggest_memory_format());

      const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_clamp_nc_u8(
          qnnpack_operator, /* clamp */
          input_contig.size(0) /* batch size */,
          (u8*)input_contig.data_ptr<quint8>() /* input data */,
          num_elems /* input stride */,
          (u8*)qy.data_ptr<quint8>() /* output data */,
          num_elems /* output stride */);
      TORCH_INTERNAL_ASSERT(
          setupStatus == pytorch_qnnp_status_success,
          "failed to setup QNNPACK Relu operator");

      pthreadpool_t threadpool = pthreadpool_();

      const pytorch_qnnp_status runStatus =
          pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

      TORCH_INTERNAL_ASSERT(
          runStatus == pytorch_qnnp_status_success,
          "failed to run QNNPACK Relu operator");
      return qy;
        */
}

pub fn relu_quantized_cpu(qx: &Tensor) -> Tensor {
    
    todo!();
        /*
            #ifdef USE_PYTORCH_QNNPACK
      if (globalContext().qEngine() == QEngine::QNNPACK && qx.scalar_type() == kQUInt8) {
        return qnnpack_relu(qx);
      }
      #endif
      Tensor qy;
      qrelu_stub(qx.device().type(), qx, qy);
      return qy;
        */
}


pub fn relu_quantized_cpu_mut(qx: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            const auto zero_point = qx.q_zero_point();
      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
        using Vec = Vectorized<Scalar>;
        auto iter = TensorIterator::unary_op(qx, qx);
        auto zero_point_vec = Vec(Scalar(zero_point));
        cpu_kernel_vec(
            iter,
            [&](Scalar value) -> Scalar {
              return Scalar(max<underlying_t>(value.val_, zero_point));
            },
            [&](Vec value) -> Vec { return value.relu(zero_point_vec); });
      });
      return qx;
        */
}

pub fn leaky_relu_out_quantized_cpu(
        self_:  &Tensor,
        negval: &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            qrelu_leaky_stub(self.device().type(), result, self, negval);
      return result;
        */
}

pub fn leaky_relu_quantized_cpu(
        self_:  &Tensor,
        negval: &Scalar) -> Tensor {
    
    todo!();
        /*
            const auto qx = self.contiguous(self.suggest_memory_format());
      auto qy = _empty_affine_quantized(qx.sizes(),
          device(kCPU).dtype(self.scalar_type()),
          qx.q_scale(),
          qx.q_zero_point(),
          self.suggest_memory_format());
      qrelu_leaky_stub(self.device().type(), qy, qx, negval);
      return qy;
        */
}

pub fn leaky_relu_quantized_cpu_mut(
        self_:  &mut Tensor,
        negval: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            qrelu_leaky_stub(self.device().type(), self, self, negval);
      return self;
        */
}

pub fn quantized_relu6(qx: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor qy;
      qrelu6_stub(qx.device().type(), qx, qy);
      return qy;
        */
}

pub fn quantized_relu6_mut(qx: &mut Tensor) -> Tensor {
    
    todo!();
        /*
            const auto zero_point = qx.q_zero_point();
      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu6_", [&]() {
        using Vec = Vectorized<Scalar>;
        auto iter = TensorIterator::unary_op(qx, qx);
        auto zero_point_vec = Vec(Scalar(zero_point));
        Scalar six = native::quantize_val<Scalar>(
            qx.q_scale(),
            qx.q_zero_point(),
            /*value=*/6.0);
        auto six_vec = Vec(six);
        cpu_kernel_vec(
            iter,
            [&](Scalar value) -> Scalar {
              underlying_t relu_val = max<underlying_t>(value.val_,
                                                             zero_point);
              return Scalar(min<underlying_t>(relu_val, six.val_));
            },
            [&](Vec value) -> Vec { return value.relu6(zero_point_vec, six_vec); });
      });
      return qx;
        */
}

pub struct QRelu6 {

}

impl QRelu6 {
    
    pub fn run(
        qx:      Tensor,
        inplace: bool) -> Tensor {
        
        todo!();
        /*
            if (inplace) {
          return quantized_relu6_(qx);
        } else {
          return quantized_relu6(qx);
        }
        */
    }
}

pub struct QLeakyRelu {

}

impl QLeakyRelu {
    
    pub fn run(
        self_:             Tensor,
        negative_slope:    &Scalar,
        inplace:           bool,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
        
        todo!();
        /*
            // inplace argument is ignored now, TODO:support inplace
        if (inplace) {
          TORCH_WARN("inplace=True is not supported for quantized::leaky_relu yet");
        }
        const auto qx = self.contiguous(self.suggest_memory_format());
        auto qy = _empty_affine_quantized(qx.sizes(),
          device(kCPU).dtype(self.scalar_type()),
          output_scale,
          output_zero_point,
          self.suggest_memory_format());
        qrelu_leaky_stub(self.device().type(), qy, qx, negative_slope);
        return qy;
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::relu6"), TORCH_FN(QRelu6::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::leaky_relu"), TORCH_FN(QLeakyRelu::run));
    }
    */
}
