crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/SegmentReduce.h]

pub enum SegmentReductionType { 
    MAX, 
    MEAN 
}

pub type SegmentReduceFn = fn(
        _0: SegmentReductionType,
        _1: &Tensor,
        _2: &Tensor,
        _3: i64,
        _4: &Option<Scalar>
) -> Tensor;

declare_dispatch!{segment_reduce_fn, _segment_reduce_stub}

pub type SegmentReduceBackwardFn = fn(
        _0: &Tensor,
        _1: &Tensor,
        _2: &Tensor,
        _3: SegmentReductionType,
        _4: &Tensor,
        _5: i64
) -> Tensor;

declare_dispatch!{segment_reduce_backward_fn, _segment_reduce_backward_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/SegmentReduce.cpp]

define_dispatch!{_segment_reduce_stub}
define_dispatch!{_segment_reduce_backward_stub}

pub fn get_reduction_enum(reduce: &StringView) -> SegmentReductionType {
    
    todo!();
        /*
            if (reduce == "max") {
        return SegmentReductionType::MAX;
      } else if (reduce == "mean") {
        return SegmentReductionType::MEAN;
      } else {
        TORCH_CHECK(false, "unsopported reduction given! ", reduce);
      }
        */
}

pub fn segment_reduce_cpu_kernel(
    reduction: SegmentReductionType,
    data:      &Tensor,
    lengths:   &Tensor,
    axis:      i64,
    initial:   &Option<Scalar>) -> Tensor {

    todo!();
        /*
            i64 segment_count = lengths.numel();
      auto output_shape = data.sizes().vec();
      output_shape[axis] = segment_count;
      auto output = empty(output_shape, data.options());

      i64 stride_count = data.numel() / data.size(axis);
      const auto* lengths_data = lengths.data_ptr<i64>();

      AT_DISPATCH_ALL_TYPES_AND2(
          kBFloat16, kHalf, data.scalar_type(), "_segment_reduce_cpu", ([&]() {
            auto* output_data = output.data_ptr<Scalar>();
            const auto* values_data = data.data_ptr<Scalar>();
            i64 lengths_cum_sum = 0;
            for (i64 i = 0; i < segment_count; ++i) {
              for (i64 l = 0; l < stride_count; ++l) {
                // ===== step1: initialize starting value
                Scalar initial_value;
                if (initial.has_value()) {
                  initial_value = initial.value().to<Scalar>();
                } else if (reduction == SegmentReductionType::MAX) {
                  initial_value = numeric_limits<Scalar>::lowest();
                } else if (reduction == SegmentReductionType::MEAN) {
                  initial_value = 0;
                }

                // ===== step2: apply reduction
                for (i64 j = 0; j < lengths_data[i]; ++j) {
                  i64 starting_index =
                      ((lengths_cum_sum + j) * stride_count) + l;
                  const auto data = values_data[starting_index];
                  // TODO: There is no need to branch with every element
                  if (reduction == SegmentReductionType::MAX) {
                    initial_value = _isnan(data)
                        ? data
                        : max<Scalar>(initial_value, data);
                  } else if (reduction == SegmentReductionType::MEAN) {
                    initial_value = initial_value + data;
                  }
                }

                // ===== step3: finalize reduction
                TORCH_CHECK(lengths_data[i] >= 0);

                if (lengths_data[i] == 0 && !initial.has_value()) {
                  initial_value = static_cast<Scalar>(NAN);
                } else if (
                    reduction == SegmentReductionType::MEAN &&
                    lengths_data[i] > 0 && !_isnan(initial_value)) {
                  initial_value = initial_value / lengths_data[i];
                }
                i64 output_index = (i * stride_count) + l;
                output_data[output_index] = initial_value;
              }
              lengths_cum_sum += lengths_data[i];
            }
          }));

      return output;
        */
}

pub fn segment_reduce_cpu_backward_kernel(
    grad_contig:    &Tensor,
    output_contig:  &Tensor,
    data_contig:    &Tensor,
    reduction:      SegmentReductionType,
    lengths_contig: &Tensor,
    axis:           i64) -> Tensor {

    todo!();
        /*
            i64 segment_count = lengths_contig.numel();
      auto output_shape = data_contig.sizes().vec();
      output_shape[axis] = segment_count;
      auto grad_input = zeros({data_contig.sizes()}, grad_contig.options());

      i64 stride_count = data_contig.numel() / data_contig.size(axis);
      const auto* lengths_data = lengths_contig.data_ptr<i64>();

      // TODO: Swtich to TensorIterator for better maintainablility and readability
      AT_DISPATCH_ALL_TYPES_AND2(
          kBFloat16,
          kHalf,
          data_contig.scalar_type(),
          "_segment_reduce_cpu",
          ([&]() {
            auto* output_data = output_contig.data_ptr<Scalar>();
            auto* grad_data = grad_contig.data_ptr<Scalar>();
            auto* grad_input_data = grad_input.data_ptr<Scalar>();
            const auto* values_data = data_contig.data_ptr<Scalar>();

            i64 lengths_cum_sum = 0;
            for (i64 i = 0; i < segment_count; ++i) {
              if (lengths_data[i] == 0) {
                continue;
              }

              for (i64 l = 0; l < stride_count; ++l) {
                i64 output_index = (i * stride_count) + l;

                if (reduction == SegmentReductionType::MAX) {
                  i64 counter = 0;
                  for (i64 j = 0; j < lengths_data[i]; ++j) {
                    i64 starting_index =
                        ((lengths_cum_sum + j) * stride_count) + l;
                    if (_isnan(values_data[starting_index]) ||
                        values_data[starting_index] == output_data[output_index]) {
                      grad_input_data[starting_index] = grad_data[output_index];
                      counter++;
                    }
                  }
                  // Average gradient based on number of maximum elements in the
                  // segment
                  if (counter < 2) {
                    continue;
                  }
                  for (i64 j = 0; j < lengths_data[i]; ++j) {
                    i64 starting_index =
                        ((lengths_cum_sum + j) * stride_count) + l;
                    if (grad_input_data[starting_index] > 0) {
                      grad_input_data[starting_index] =
                          grad_input_data[starting_index] / counter;
                    }
                  }
                } else if (reduction == SegmentReductionType::MEAN) {
                  auto grad_val = grad_data[output_index] / lengths_data[i];
                  for (i64 j = 0; j < lengths_data[i]; ++j) {
                    i64 starting_index =
                        ((lengths_cum_sum + j) * stride_count) + l;
                    grad_input_data[starting_index] = grad_val;
                  }
                }
              }

              lengths_cum_sum += lengths_data[i];
            }
          }));

      return grad_input;
        */
}

pub fn segment_reduce_kernel(
    data:    &Tensor,
    reduce:  StringView,
    lengths: &Option<Tensor>,
    indices: &Option<Tensor>,
    axis:    i64,
    unsafe_: bool,
    initial: &Option<Scalar>) -> Tensor {

    todo!();
    /*
            axis = maybe_wrap_dim(axis, data.ndimension());
      TORCH_CHECK(axis == 0, "Currently only dim=0 is supported! ", axis);
      TORCH_CHECK(data.numel() > 0);

      // length related checks
      TORCH_CHECK(
          lengths.has_value() && !indices.has_value(),
          "Currently only lengths based reduction is supported!")
      const auto& lengths_value = lengths.value();
      TORCH_CHECK(lengths_value.dim() == 1);
      TORCH_CHECK(data.get_device() == lengths_value.get_device());
      TORCH_CHECK(data.dim() >= lengths_value.dim());

      if (!unsafe) {
        auto min_length = lengths_value.min().item<i64>();
        TORCH_CHECK((min_length >= 0), "lengths contains negative value!");
        TORCH_CHECK(min_length != 0 || initial.has_value());
        TORCH_CHECK(lengths_value.sum().item<i64>() == data.size(axis));
      }

      auto reduction = get_reduction_enum(reduce);
      const auto data_contig = data.contiguous();
      const auto lengths_contig = lengths_value.contiguous();

      return _segment_reduce_stub(
          data_contig.device().type(),
          reduction,
          data_contig,
          lengths_contig,
          axis,
          initial);
        */
}

register_arch_dispatch!(
    _segment_reduce_stub,
    DEFAULT,
    &_segment_reduce_cpu_kernel
);

register_avx_dispatch!{_segment_reduce_stub, &_segment_reduce_cpu_kernel}
register_avx2_dispatch!{_segment_reduce_stub, &_segment_reduce_cpu_kernel}
register_vsx_dispatch!{_segment_reduce_stub, &_segment_reduce_cpu_kernel}

/**
  | Currently some computation is beind duplicated
  | across forward and backward.
  |
  | TODO: Cache indices in forward pass to re-use
  | in backward
  |
  */
pub fn segment_reduce_backward_kernel(
    grad:    &Tensor,
    output:  &Tensor,
    data:    &Tensor,
    reduce:  StringView,
    lengths: &Option<Tensor>,
    axis:    i64) -> Tensor {

    todo!();
        /*
            axis = maybe_wrap_dim(axis, data.ndimension());
      TORCH_CHECK(axis == 0, "Currently only dim=0 is supported! ", axis);
      TORCH_CHECK(
          lengths.has_value(),
          "Currently only lengths based reduction is supported!")
      const auto& lengths_value = lengths.value();

      const auto grad_contig = grad.contiguous();
      const auto output_contig = output.contiguous();
      const auto data_contig = data.contiguous();
      const auto lengths_contig = lengths_value.contiguous();

      auto reduction = get_reduction_enum(reduce);
      return _segment_reduce_backward_stub(
          grad_contig.device().type(),
          grad_contig,
          output_contig,
          data_contig,
          reduction,
          lengths_contig,
          axis);
        */
}

register_arch_dispatch!(
    _segment_reduce_backward_stub,
    DEFAULT,
    &_segment_reduce_cpu_backward_kernel
);

register_avx_dispatch!(
    _segment_reduce_backward_stub,
    &_segment_reduce_cpu_backward_kernel
);

register_avx2_dispatch!(
    _segment_reduce_backward_stub,
    &_segment_reduce_cpu_backward_kernel
);

register_vsx_dispatch!(
    _segment_reduce_backward_stub,
    &_segment_reduce_cpu_backward_kernel
);
