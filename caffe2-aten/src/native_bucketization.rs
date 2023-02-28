/*! 
 | Implement a TF like searchsorted and
 | a bucketize function running on cpu
 |
 | - torch.searchsorted(sorted_sequence, values,
 | right=False, out_int32=False)
 |
 |   sorted_sequence - N*D or 1D (apply to all
 |   values) tensor containing sorted sequences in
 |   last dimension
 |
 |   values          - N*D tensor or a Scalar
 |   (when sorted_sequence is 1D) containing the
 |   search values
 |
 |   right           - corresponding to lower
 |   bound if False and upper bound if True
 |
 |   out_int32       - the output tensor is
 |   i64 type if False and int(32bit normally)
 |   type if True.
 |
 | - torch.bucketize(values, boundaries,
 | right=False, out_int32=False)
 |
 |   values     - N*D tensor or a Scalar
 |   containing the search value
 |
 |   boundaries - 1D tensor containing a sorted
 |   sequences
 |
 |   right      - corresponding to lower bound if
 |   False and upper bound if True
 |
 |   out_int32  - the output tensor is i64
 |   type if False and int(32bit normally) type if
 |   True.
 |
 | - Restrictions are defined in
 | searchsorted_pre_check()
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Bucketization.cpp]

/// minimal size for searchsorted_cpu_contiguous
/// to run parallel (multithread)
///
pub const SEARCHSORTED_GRAIN_SIZE: i64 = 200;

/**
  | customized lower_bound func to ensure the low
  | bound of 'nan', 'inf' etc. be the end of
  | boundary lower_bound can not be used here since
  | its customized comparator need strict weak
  | ordering
  |
  */
pub fn cus_lower_bound<input_t>(
        start: *const Input,
        end:   *const Input,
        val:   Input) -> *const Input {

    todo!();
        /*
            while (start < end) {
        const input_t* mid = start + ((end - start) >> 1);
        if (!(*mid >= val)) {
          start = mid + 1;
        }
        else {
          end = mid;
        }
      }
      return start;
        */
}

pub fn searchsorted_cpu_contiguous<input_t, output_t>(
        result:     &mut Tensor,
        input:      &Tensor,
        boundaries: &Tensor,
        right:      &bool)  {

    todo!();
        /*
            i64 numel_in = input.numel();
      bool is_scalar_input = input.dim() == 0 && numel_in == 1;
      // inner most dim size of input and boundaries
      i64 idim_in = is_scalar_input ? 1 : input.sizes().back();
      i64 idim_bd = boundaries.sizes().back();

      const input_t *data_in = input.data_ptr<input_t>();
      const input_t *data_bd = boundaries.data_ptr<input_t>();
      output_t *data_out = result.data_ptr<output_t>();

      bool is_1d_boundaries = boundaries.dim() == 1;
      parallel_for(0, numel_in, SEARCHSORTED_GRAIN_SIZE, [&](i64 start, i64 end) {
        for (i64 i = start; i < end; ++i) {
          // If boundaries tensor is 1d, we always search the entire boundary tensor
          i64 start_bd = is_1d_boundaries ? 0 : i / idim_in * idim_bd;
          const input_t *data_bd_start = &data_bd[start_bd];

          i64 pos = !right ?
            cus_lower_bound(data_bd_start, data_bd_start + idim_bd, data_in[i]) - data_bd_start :
            upper_bound(data_bd_start, data_bd_start + idim_bd, data_in[i]) - data_bd_start;

          // type conversion might happen here
          data_out[i] = pos;
        }
      });
        */
}

pub fn dispatch(
        result:     &mut Tensor,
        input:      &Tensor,
        boundaries: &Tensor,
        out_int32:  bool,
        right:      bool)  {
    
    todo!();
        /*
            if (!out_int32) {
        AT_DISPATCH_ALL_TYPES(input.scalar_type(), "searchsorted_out_cpu", [&] {
          searchsorted_cpu_contiguous<Scalar, i64>(result, input, boundaries, right);
        });
      }
      else {
        AT_DISPATCH_ALL_TYPES(input.scalar_type(), "searchsorted_out_cpu", [&] {
          searchsorted_cpu_contiguous<Scalar, int>(result, input, boundaries, right);
        });
      }
        */
}

pub fn searchsorted_out_cpu(
        sorted_sequence: &Tensor,
        self_:           &Tensor,
        out_int32:       bool,
        right:           bool,
        result:          &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            searchsorted_pre_check(sorted_sequence, self, result, out_int32);
      if (result.numel() == 0) {
        result.resize_(self.sizes());
      }
      if (self.numel() == 0) {
        return result;
      }
      if (sorted_sequence.is_contiguous() && self.is_contiguous() && sorted_sequence.dtype() == self.dtype()) {
        dispatch(result, self, sorted_sequence, out_int32, right);
        return result;
      }

      Tensor trimmed_input;
      Tensor trimmed_boundaries;
      searchsorted_maybe_trim_input_tensors(trimmed_input, trimmed_boundaries, self, sorted_sequence);
      const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
      const Tensor& final_boundaries = trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
      dispatch(result, final_input, final_boundaries, out_int32, right);
      return result;
        */
}

pub fn searchsorted_cpu_tensor_tensor(
        sorted_sequence: &Tensor,
        self_:           &Tensor,
        out_int32:       bool,
        right:           bool) -> Tensor {
    
    todo!();
        /*
            ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
      TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
      Tensor result = empty({0}, options, MemoryFormat::Contiguous);
      native::searchsorted_out_cpu(sorted_sequence, self, out_int32, right, result);
      return result;
        */
}

pub fn searchsorted_cpu_tensor_scalar(
        sorted_sequence: &Tensor,
        self_:           &Scalar,
        out_int32:       bool,
        right:           bool) -> Tensor {
    
    todo!();
        /*
            return searchsorted_cpu(sorted_sequence, searchsorted_scalar_tensor(self, sorted_sequence.device()), out_int32, right);
        */
}

pub fn bucketize_out_cpu(
        self_:      &Tensor,
        boundaries: &Tensor,
        out_int32:  bool,
        right:      bool,
        result:     &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
      native::searchsorted_out_cpu(boundaries, self, out_int32, right, result);
      return result;
        */
}

pub fn bucketize_cpu_tensor_tensor(
        self_:      &Tensor,
        boundaries: &Tensor,
        out_int32:  bool,
        right:      bool) -> Tensor {
    
    todo!();
        /*
            ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
      TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
      Tensor result = empty({0}, options, MemoryFormat::Contiguous);
      native::bucketize_out_cpu(self, boundaries, out_int32, right, result);
      return result;
        */
}

pub fn bucketize_cpu_scalar_tensor(
        self_:      &Scalar,
        boundaries: &Tensor,
        out_int32:  bool,
        right:      bool) -> Tensor {
    
    todo!();
        /*
            return bucketize_cpu(searchsorted_scalar_tensor(self, boundaries.device()), boundaries, out_int32, right);
        */
}
