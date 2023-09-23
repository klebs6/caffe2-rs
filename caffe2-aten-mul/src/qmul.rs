crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qmul.cpp]

define_dispatch!{qmul_relu_stub}
define_dispatch!{qmul_stub}

#[inline] pub fn check_inputs(
        qa: &Tensor,
        qb: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(qa.qscheme() == kPerTensorAffine,
                  "Only per tensor quantization is supported in Mul.");
      TORCH_CHECK(qa.scalar_type() == qb.scalar_type(),
                  "Mul operands should have same data type.");
      TORCH_CHECK(qa.qscheme() == qb.qscheme(),
                  "Both inputs to Mul must have the same quantization shceme.");
        */
}

/**
  | Note: out is assumed to be the same size as
  | self and other.
  |
  | Note: Multiplication is only supported when
  |       self, other, out are of the same dtype.
  */
pub fn mul_out<const ReLUFused: bool = false>(
        out:   &mut Tensor,
        self_: &Tensor,
        other: &Tensor) -> Tensor {

    todo!();
        /*
            if (ReLUFused) {
        qmul_relu_stub(self.device().type(), out, self, other);
      } else {
        qmul_stub(self.device().type(), out, self, other);
      }
      return out;
        */
}

pub fn mul_scalar_out<const ReLUFused: bool = false>(
        out:   &mut Tensor,
        self_: &Tensor,
        other: &Scalar) -> Tensor {

    todo!();
        /*
            i64 self_zero_point = self.q_zero_point();
      double self_scale = self.q_scale();
      double other_val = other.toDouble();

      double scale_prime;
      i64 zero_point_prime;

      AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qmul_scalar", [&]() {
        i64 q_min = underlying_t::min;
        i64 q_max = underlying_t::max;

        if (other_val > 0.0) {
          scale_prime = other_val * self_scale;
          zero_point_prime = self_zero_point;

          if (ReLUFused) {
            qrelu_stub(self.device().type(), self, out);
          } else {
            out.copy_(self);
          }
          set_quantizer_(out, make_per_tensor_affine_quantizer(
              scale_prime, zero_point_prime, self.scalar_type()));
        } else if (other_val == 0.0) {
          scale_prime = 1.0;
          zero_point_prime = 0;

          // Strided "memset"
          // Set all values to 0
          auto iter = TensorIterator::unary_op(out, self);
          cpu_kernel_vec(
              iter,
              [&](Scalar a) -> Scalar { return Scalar(0); },
              [&](Vectorized<Scalar> vec) -> Vectorized<Scalar> {
                return Vectorized<Scalar>(Scalar(0));
              });
          set_quantizer_(out, make_per_tensor_affine_quantizer(
              scale_prime, zero_point_prime, self.scalar_type()));
        } else /* other_val < 0.0 */ {
          scale_prime = abs(other_val) * self_scale;
          zero_point_prime = q_max - (self_zero_point - q_min);

          // xq' = q_max + q_min - x_q
          auto iter = TensorIterator::unary_op(out, self);
          cpu_kernel(
              iter,
              [&](Scalar a) -> Scalar {
                a = Scalar(underlying_t(q_max + q_min - a.val_));
                if (ReLUFused) {
                  a = Scalar(max(a.val_, underlying_t(zero_point_prime)));
                }
                return a;
              });
          set_quantizer_(out, make_per_tensor_affine_quantizer(
              scale_prime, zero_point_prime, self.scalar_type()));
        }
      });

      return out;
        */
}

pub struct QMul<const ReLUFused: bool = false> {

}

impl<const ReLUFused: bool> QMul<ReLUFused> {
    
    pub fn run(
        qa:         Tensor,
        qb:         Tensor,
        scale:      f64,
        zero_point: i64) -> Tensor {
        
        todo!();
        /*
            check_inputs(qa, qb);
        auto qc = _empty_affine_quantized(
            infer_size_dimvector(qa.sizes(), qb.sizes()),
            device(kCPU).dtype(qa.scalar_type()),
            scale,
            zero_point,
            qa.suggest_memory_format());
        return _mul_out<ReLUFused>(qc, qa, qb);
        */
    }
}

pub struct QMulOut<const ReLUFused: bool = false> {

}

impl<const ReLUFused: bool> QMulOut<ReLUFused> {
    
    pub fn run(
        qa:  Tensor,
        qb:  Tensor,
        out: Tensor) -> Tensor {
        
        todo!();
        /*
            check_inputs(qa, qb);
        return _mul_out<ReLUFused>(out, qa, qb);
        */
    }
}

pub struct QMulScalar<const ReLUFused: bool = false> {

}

impl<const ReLUFused: bool> QMulScalar<ReLUFused> {
    
    pub fn run(
        qa: Tensor,
        b:  &Scalar) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
                  qa.qscheme() == kPerTensorSymmetric,
                  "Only per tensor quantization is supported in Mul.");
        auto qc = empty_like(qa, qa.suggest_memory_format());
        return _mul_scalar_out<ReLUFused>(qc, qa, b);
        */
    }
}

pub struct QMulScalar2<const ReLUFused: bool = false> {

}

impl<const ReLUFused: bool> QMulScalar2<ReLUFused> {
    
    pub fn run(
        b:  &Scalar,
        qa: Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
                  qa.qscheme() == kPerTensorSymmetric,
                  "Only per tensor quantization is supported in Mul.");
        auto qc = empty_like(qa, qa.suggest_memory_format());
        return _mul_scalar_out<ReLUFused>(qc, qa, b);
        */
    }
}

pub struct QMulScalarOut<const ReLUFused: bool = false> {

}

impl<const ReLUFused: bool> QMulScalarOut<ReLUFused> {
    
    pub fn run(
        qa:  Tensor,
        b:   &Scalar,
        out: Tensor) -> Tensor {
        
        todo!();
        /*
            check_inputs(qa, out);
        return _mul_scalar_out<ReLUFused>(out, qa, b);
        */
    }
}

/**
  | `torch.jit.trace` will trace Scalar as Tensor
  |
  | This can be removed after broadcast is
  | supported and all variations of
  | `quantized::mul` is merged into
  | `quantized::mul`
  */
pub struct QMulScalarTensor<const ReLUFused: bool = false> {

}

impl<const ReLUFused: bool> QMulScalarTensor<ReLUFused> {
    
    pub fn run(
        qa: Tensor,
        b:  Tensor) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
                  qa.qscheme() == kPerTensorSymmetric,
                  "Only per tensor quantization is suported in Mul.");
        auto qc = empty_like(qa, qa.suggest_memory_format());
        return _mul_scalar_out<ReLUFused>(qc, qa, b.item());
        */
    }
}

/**
  | `torch.jit.trace` will trace Scalar as Tensor
  |
  | This can be removed after broadcast is
  | supported and all variations of
  | `quantized::mul` is merged into
  | `quantized::mul`
  */
pub struct QMulScalarTensorOut<const ReLUFused: bool = false> {

}

impl<const ReLUFused: bool> QMulScalarTensorOut<ReLUFused> {
    
    pub fn run(
        qa:  Tensor,
        b:   Tensor,
        out: Tensor) -> Tensor {
        
        todo!();
        /*
            check_inputs(qa, out);
        return _mul_scalar_out<ReLUFused>(out, qa, b.item());
        */
    }
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul"),                 TORCH_FN(QMul</*ReLUFused=*/false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul.out"),             TORCH_FN(QMulOut</*ReLUFused=*/false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul.Scalar"),          TORCH_FN(QMulScalar</*ReLUFused=*/false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul.Scalar2"),          TORCH_FN(QMulScalar2</*ReLUFused=*/false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul.Scalar_out"),      TORCH_FN(QMulScalarOut</*ReLUFused=*/false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu"),            TORCH_FN(QMul</*ReLUFused=*/true>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.out"),        TORCH_FN(QMulOut</*ReLUFused=*/true>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.Scalar"),     TORCH_FN(QMulScalar</*ReLUFused=*/true>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.Scalar2"),     TORCH_FN(QMulScalar2</*ReLUFused=*/true>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.Scalar_out"), TORCH_FN(QMulScalarOut</*ReLUFused=*/true>::run));
      // deprecated functions, kept for backward compatibility
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_out"),             TORCH_FN(QMulOut</*ReLUFused=*/false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu_out"),        TORCH_FN(QMulOut</*ReLUFused=*/true>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar"),          TORCH_FN(QMulScalar</*ReLUFused=*/false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu"),     TORCH_FN(QMulScalar</*ReLUFused=*/true>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_out"),      TORCH_FN(QMulScalarOut</*ReLUFused=*/false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu_out"), TORCH_FN(QMulScalarOut</*ReLUFused=*/true>::run));
      // TODO: remove after broadcasting is supported
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar.Tensor"), TORCH_FN(QMulScalarTensor</*ReLUFused=*/false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu.Tensor"), TORCH_FN(QMulScalarTensor</*ReLUFused=*/true>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_out.Tensor"), TORCH_FN(QMulScalarTensorOut</*ReLUFused=*/false>::run));
      m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu_out.Tensor"), TORCH_FN(QMulScalarTensorOut</*ReLUFused=*/true>::run));
    }
    */
}
