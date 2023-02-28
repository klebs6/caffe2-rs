crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/UpSampleKernel.cpp]

pub type scale_t = Vec<Option<f64>>;

#[inline] pub fn nearest_idx(
        output_index: i64,
        input_size:   i64,
        output_size:  i64,
        scales:       Option<f64>) -> i64 {
    
    todo!();
        /*
            if (output_size == input_size) {
        // scale_factor = 1, simply copy
        return output_index;
      } else if (output_size == 2 * input_size) {
        // scale_factor = 2, shift input index
        return output_index >> 1;
      } else {
        float scale = compute_scales_value<float>(scales, input_size, output_size);
        return nearest_neighbor_compute_source_index(scale, output_index, input_size);
      }
        */
}

/**
  | Helper structs and methods for
  | cpu_upsample_linear
  |
  | Interpolation methods that used below are
  | separable, and as such we can compute the
  | interpolation independently per dimension in
  | a recursive way. Please, refer to #10482 for
  | more context.
  |
  | Linear Interpolation structure to compute
  | output value in n-dimensional case.
  |
  | - recursively compute interpolated output for
  | each dimension
  |
  | - we rely a lot on compiler's code optimization
  |   such that implemented operations can be
  |   automatically factorized and vectorized using
  |   SSE and AVX2
  |
  */
pub struct Interpolate<const n: i32,Scalar,Index,const interp_size: i32> {

}

impl<const n: i32,Scalar,Index,const INTERP_SIZE: i32> Interpolate<n,Scalar,Index,INTERP_SIZE> {
    
    #[inline] pub fn eval(
        src:     *mut u8,
        data:    *mut *mut u8,
        strides: *const i64,
        i:       i64) -> Scalar {
        
        todo!();
        /*
            Index ids = *(Index*)&data[0][i * strides[0]];
          Scalar wts = *(Scalar*)&data[1][i * strides[1]];
          Scalar t = Interpolate<n - 1, Scalar, Index, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);
          Scalar output = t * wts;
          for (int j=1; j<interp_size; j++) {
            ids = *(Index*)&data[2 * j + 0][i * strides[2 * j + 0]];
            wts = *(Scalar*)&data[2 * j + 1][i * strides[2 * j + 1]];
            t = Interpolate<n - 1, Scalar, Index, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);
            output += t * wts;
          }
          return output;
        */
    }
}

//template <typename Scalar, typename Index, int interp_size>
pub struct Interpolate1<Scalar, Index, interp_size> {

}

impl Interpolate1<Scalar, Index, interp_size> {
    
    #[inline] pub fn eval(
        src:     *mut u8,
        data:    *mut *mut u8,
        strides: *const i64,
        i:       i64) -> Scalar {
        
        todo!();
        /*
            Index ids = *(Index*)&data[0][i * strides[0]];
          Scalar wts = *(Scalar*)&data[1][i * strides[1]];
          Scalar t = *(Scalar *)&src[ids];
          Scalar output = t * wts;
          for (int j=1; j<interp_size; j++) {
            ids = *(Index*)&data[2 * j + 0][i * strides[2 * j + 0]];
            wts = *(Scalar*)&data[2 * j + 1][i * strides[2 * j + 1]];
            t = *(Scalar *)&src[ids];
            output += t * wts;
          }
          return output;
        */
    }
}

pub struct InterpolateNInterp1<const n: i32,Scalar,Index> {

}

impl<const n: i32,Scalar,Index> InterpolateNInterp1<n,Scalar,Index> {
    
    #[inline] pub fn eval(
        src:     *mut u8,
        data:    *mut *mut u8,
        strides: *const i64,
        i:       i64) -> Scalar {
        
        todo!();
        /*
            Index ids = *(Index*)&data[0][i * strides[0]];
          return Interpolate<n - 1, Scalar, Index, 1>::eval(src + ids, &data[2], &strides[2], i);
        */
    }
}

pub struct Interpolate1Interp1<Scalar,Index> {

}

impl Interpolate1Interp1<Scalar,Index> {
    
    #[inline] pub fn eval(
        src:     *mut u8,
        data:    *mut *mut u8,
        strides: *const i64,
        i:       i64) -> Scalar {
        
        todo!();
        /*
            Index ids = *(Index*)&data[0][i * strides[0]];
          return *(Scalar *)&src[ids];
        */
    }
}

// There is an unexpected 2x slowdown for upsample_trilinear3d channels_first
// for both 1 and 6 threads. We have to specialize this case as below:
// Once the issue is fixed we can keep generic implementation and remove:
// struct Interpolate<n, Scalar, Index, 2> and
// struct Interpolate<1, Scalar, Index, 2>
pub struct InterpolateNInterp2<const N: i32,Scalar,Index> {

}

impl<const N: i32,Scalar,Index> InterpolateNInterp2<N,Scalar,Index> {

    #[inline] pub fn eval(
        src:     *mut u8,
        data:    *mut *mut u8,
        strides: *const i64,
        i:       i64) -> Scalar {
        
        todo!();
        /*
            Index i0 = *(Index*)&data[0][i * strides[0]];
            Index i1 = *(Index*)&data[2][i * strides[2]];
            Scalar w0 = *(Scalar *)&data[1][i * strides[1]];
            Scalar w1 = *(Scalar *)&data[3][i * strides[3]];

            Scalar t0 = Interpolate<n - 1, Scalar, Index, 2>::eval(src + i0, &data[4], &strides[4], i);
            Scalar t1 = Interpolate<n - 1, Scalar, Index, 2>::eval(src + i1, &data[4], &strides[4], i);

            return t0 * w0 + t1 * w1;
        */
    }
}

pub struct Interpolate1Interp2<Scalar,Index> {

}

impl Interpolate1Interp2<Scalar,Index> {
    
    #[inline] pub fn eval(
        src:     *mut u8,
        data:    *mut *mut u8,
        strides: *const i64,
        i:       i64) -> Scalar {
        
        todo!();
        /*
            Index i0 = *(Index*)&data[0][i * strides[0]];
            Index i1 = *(Index*)&data[2][i * strides[2]];
            Scalar w0 = *(Scalar *)&data[1][i * strides[1]];
            Scalar w1 = *(Scalar *)&data[3][i * strides[3]];
            Scalar t0 = *(Scalar *)&src[i0];
            Scalar t1 = *(Scalar *)&src[i1];
            return t0 * w0 + t1 * w1;
        */
    }
}

#[inline] pub fn interpolate<const N: i32, Scalar, Index, const interp_size: i32>(
        src:     *mut u8,
        data:    *mut *mut u8,
        strides: *const i64,
        i:       i64) -> Scalar {

    todo!();
        /*
            return Interpolate<n, Scalar, Index, interp_size>::eval(src, data, strides, i);
        */
}

#[inline] pub fn is_zero_stride<const interp_size: i32>(strides: *const i64) -> bool {

    todo!();
        /*
            bool output = strides[0] == 0;
      for (int i=1; i<2 * interp_size; i++) {
        output &= (strides[i] == 0);
      }
      return output;
        */
}

#[inline] pub fn is_contiguous_stride<Scalar, Index, const interp_size: i32>(strides: *const i64) -> bool {

    todo!();
        /*
            bool output = (strides[0] == sizeof(Index)) && (strides[1] == sizeof(Scalar));
      for (int i=2; i<2 * interp_size; i+=2) {
        output &= (strides[i] == sizeof(Index)) && (strides[i + 1] == sizeof(Scalar));
      }
      return output;
        */
}

// Helper class to recursively check if all input strides corresponding to interpolated dimensions
// are equal zero except on a single dimension.
//
// Inputs: array of strides of size N, non_zero_stride_dim which can be -1, 0, 1, 2, ...
//   if non_zero_stride_dim, we check that all strides are equal zero, otherwise
//   4 strides corresponding to the strides for index_0, weight_0, index_1 and weight_1 for non_zero_stride_dim
//   dimension should be non zero.
//
// Unit check of the recursion is to verify whether 4 strides for one interpolated dimension are either zero,
// see method is_zero_stride, or (sizeof(Index), sizeof(Scalar), sizeof(Index), sizeof(Scalar)), see
// method is_contiguous_stride.
//
// In practice, we have the following cases:
// - for ND, float32, channel first, strides are
//         dimN-1,              dim1,           dim0
//         i0, w0, i1, w1, ..., i0, w0, i1, w1, i0, w0, i1, w1
// strides=(0,  0,  0,  0, ...,  0,  0,  0,  0,  4,  4,  4,  4)
//
// if size dim0 is 1 then its strides are 0 and dim1 strides are equal 4
//
// - for ND, float32, channel last, strides are
//         dimN-1,         dimN-2,             dim0
//         i0, w0, i1, w1, i0, w0, i1, w1, ... i0, w0, i1, w1
// strides=(0,  0,  0,  0,  0,  0,  0,  0, ..., 0,  0,  0,  0)
//
// Using these methods we can hint the compiler to factorize constant indices and weights
// in cpu_upsample_linear method
pub struct CheckAlmostAllZeroStrides<const N: i32,const non_zero_stride_dim: i32,Scalar,Index,const interp_size: i32> {

}

impl<const N: i32,const non_zero_stride_dim: i32,Scalar,Index,const interp_size: i32> 
CheckAlmostAllZeroStrides<N,non_zero_stride_dim,Scalar,Index,interp_size> {
    
    #[inline] pub fn eval(strides: *const i64) -> bool {
        
        todo!();
        /*
            // N is dim index: N -> dim0, N-1 -> dim1, ...
        // non_zero_stride_dim should be out_dims - dim
        bool output;
        if (N == non_zero_stride_dim) {
          output = is_contiguous_stride<Scalar, Index, interp_size>(strides);
        } else {
          output = is_zero_stride<interp_size>(strides);
        }
        return output &&
          CheckAlmostAllZeroStrides<N - 1, non_zero_stride_dim, Scalar, Index, interp_size>::eval(
            &strides[2 * interp_size]);
        */
    }
}

pub struct CheckAlmostAllZeroStrides0<const non_zero_stride_dim: i32,Scalar,Index,const interp_size: i32> {

}

impl<const non_zero_stride_dim: i32,Scalar,Index,const interp_size: i32> 
CheckAlmostAllZeroStrides0<non_zero_stride_dim,Scalar,Index,interp_size> {
    
    #[inline] pub fn eval(strides: *const i64) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}

#[inline] pub fn check_almost_all_zero_stride<const n: i32, const s: i32, Scalar, Index, const interp_size: i32>(strides: *const i64) -> bool {

    todo!();
        /*
            return CheckAlmostAllZeroStrides<n, s, Scalar, Index, interp_size>::eval(strides);
        */
}

/**
  | Helper method to compute interpolation
  | for nearest, linear, cubic modes
  |
  */
#[inline] pub fn basic_loop<Scalar, Index, const out_ndims: i32, const interp_size: i32>(
        data:    *mut *mut u8,
        strides: *const i64,
        n:       i64)  {

    todo!();
        /*
            char* dst = data[0];
      char* src = data[1];
      for (i64 i = 0; i < n; i++) {
        *(Scalar*)&dst[i * strides[0]] = interpolate<out_ndims, Scalar, Index, interp_size>(
            src + i * strides[1], &data[2], &strides[2], i);
      }
        */
}

/**
  | Generic upsampling computation method using
  | TensorIterator for Nd case.
  |
  | Supports: nearest, linear, cubic modes with
  | interp_size template argument: 1, 2, 4
  |
  | Single loop function for 1d, 2d and 3d cases
  | and modes
  |
  | For N dimensions, output value up to Di
  | dimension can be computed as
  |
  | output_i[a] = interpolate(output_{i+1}[a], w_{i+1}[a], output_{i+1}[a+1], w_{i+1}[a+1], ...)
  |
  | with
  |
  | output_DN[a] = interpolate(input_DN[a],
  | w_DN[a], input_DN[a+1], w_DN[a+1], ...)
  |
  | and i - dimension index and a - linear index
  | for spatial coordinates
  |
  | The recursive call is implemented with
  | InterpLinear struct using template for the loop
  | unrolling on compile time.
  |
  */
pub fn cpu_upsample_generic<Scalar, const out_ndims: i32, const interp_size: i32>(iter: &mut TensorIterator)  {

    todo!();
        /*
            auto loop = [&](char** data, const i64* strides, i64 n) {
        // special-cases to let the compiler apply compile-time input-specific optimizations
        if ((strides[0] == sizeof(Scalar) && (strides[1] == 0) &&
            check_almost_all_zero_stride<out_ndims, 1, Scalar, i64, interp_size>(&strides[2]))) {
          // contiguous channels-first case
          basic_loop<Scalar, i64, out_ndims, interp_size>(data, strides, n);
        } else if ((strides[0] == sizeof(Scalar) && (strides[1] == sizeof(Scalar)) &&
                   check_almost_all_zero_stride<out_ndims, -1, Scalar, i64, interp_size>(&strides[2]))) {
          // contiguous channels-last case
          basic_loop<Scalar, i64, out_ndims, interp_size>(data, strides, n);
        } else {
          // fallback
          basic_loop<Scalar, i64, out_ndims, interp_size>(data, strides, n);
        }
      };
      iter.for_each(loop);
        */
}

pub fn cpu_upsample_nearest_channels_last<Scalar, scale_type>(
        output: &Tensor,
        input:  &Tensor,
        scales: &ScaleType)  {

    todo!();
        /*
            TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
                  " for `output` but got dtype ", output_.dtype());

      auto input_sizes = input_.sizes().vec();
      auto output_sizes = output_.sizes().vec();
      auto ndim = input_sizes.size();
      TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

      auto channels_last_memory_format = ndim == 4 ? MemoryFormat::ChannelsLast : MemoryFormat::ChannelsLast3d;
      auto input = input_.contiguous(channels_last_memory_format);
      auto output = output_.contiguous(channels_last_memory_format);

      auto input_data = input.data_ptr<Scalar>();
      auto output_data = output.data_ptr<Scalar>();

      i64 num_batches =  input_sizes[0];
      i64 channels =  input_sizes[1];
      i64 input_depth = (ndim == 5) ? input_sizes[2] : 1;
      i64 output_depth = (ndim == 5) ? output_sizes[2] : 1;
      i64 input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
      i64 output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
      i64 input_width = input_sizes[ndim - 1];
      i64 output_width = output_sizes[ndim - 1];
      i64 numel = output.numel();

      TORCH_CHECK(channels > 0, "expected input and output channels greater than 0 but got ", channels);

      using Vec = vec::Vectorized<Scalar>;
      auto copy = [](Scalar* out, Scalar* in, i64 size) {
        i64 d = 0;
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          Vec out_vec = Vec::loadu(in + d);
          out_vec.store(out + d);
        }
        for (; d < size; d++) {
          out[d] = in[d];
        }
      };

      auto loop2d = [&](i64 begin, i64 end) {
        i64 n = 0;
        i64 oh = 0;
        i64 ow = 0;
        data_index_init(begin, n, num_batches, oh, output_height, ow, output_width);

        for (i64 i = begin; i < end; i++) {
          i64 ih = nearest_idx(oh, input_height, output_height, scales[0]);
          i64 iw = nearest_idx(ow, input_width, output_width, scales[1]);
          Scalar* output_ptr = output_data + i * channels;
          Scalar* input_ptr = input_data + n * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;
          copy(output_ptr, input_ptr, channels);
          data_index_step(n, num_batches, oh, output_height, ow, output_width);
        }
      };

      auto loop3d = [&](i64 begin, i64 end) {
        i64 n = 0;
        i64 od = 0;
        i64 oh = 0;
        i64 ow = 0;
        data_index_init(begin, n, num_batches, od, output_depth, oh, output_height, ow, output_width);

        for (i64 i = begin; i < end; i++) {
          i64 id = nearest_idx(od, input_depth, output_depth, scales[0]);
          i64 ih = nearest_idx(oh, input_height, output_height, scales[1]);
          i64 iw = nearest_idx(ow, input_width, output_width, scales[2]);
          Scalar* output_ptr = output_data + i * channels;
          Scalar* input_ptr = input_data + n * input_depth * input_height * input_width * channels +
              id * input_height * input_width * channels +
              ih * input_width * channels + iw * channels;
          copy(output_ptr, input_ptr, channels);
          data_index_step(n, num_batches, od, output_depth, oh, output_height, ow, output_width);
        }
      };

      if (ndim == 4) {
        // upsample nearest 2d
        parallel_for(0, numel / channels, internal::GRAIN_SIZE / channels, loop2d);
      } else {
        // upsample nearest 3d
        TORCH_INTERNAL_ASSERT(ndim == 5);
        parallel_for(0, numel / channels, internal::GRAIN_SIZE / channels, loop3d);
      }

      if (!output_.is_contiguous(channels_last_memory_format)) {
        output_.copy_(output);
      }
        */
}

pub fn cpu_upsample_linear_channels_last<Scalar, scale_type>(
    output:        &Tensor,
    input:         &Tensor,
    align_corners: bool,
    scales:        &ScaleType)  {

    todo!();
        /*
            TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
                  " for `output` but got dtype ", output_.dtype());

      auto input_sizes = input_.sizes().vec();
      auto output_sizes = output_.sizes().vec();
      auto ndim = input_sizes.size();
      TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

      auto channels_last_memory_format = ndim == 4 ? MemoryFormat::ChannelsLast : MemoryFormat::ChannelsLast3d;
      auto input = input_.contiguous(channels_last_memory_format);
      auto output = output_.contiguous(channels_last_memory_format);

      auto input_data = input.data_ptr<Scalar>();
      auto output_data = output.data_ptr<Scalar>();

      i64 num_batches =  input_sizes[0];
      i64 channels =  input_sizes[1];
      i64 input_depth = (ndim == 5) ? input_sizes[2] : 1;
      i64 output_depth = (ndim == 5) ? output_sizes[2] : 1;
      i64 input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
      i64 output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
      i64 input_width = input_sizes[ndim - 1];
      i64 output_width = output_sizes[ndim - 1];

      TORCH_CHECK(channels > 0, "expected input and output channels greater than 0 but got ", channels);
      i64 output_slice_size = output_depth * output_height * output_width * channels;

      using Vec = vec::Vectorized<Scalar>;
      auto loop2d = [&](i64 begin, i64 end) {
        const Scalar height_scale = area_pixel_compute_scale<Scalar>(
            input_height, output_height, align_corners, scales[0]);
        const Scalar width_scale = area_pixel_compute_scale<Scalar>(
            input_width, output_width, align_corners, scales[1]);

        auto input_indexr = [=](i64 n, i64 h, i64 w) {
          return input_data + n * input_height * input_width * channels +
              h * input_width * channels + w * channels;
        };

        i64 ih0, ih1, iw0, iw1;
        Scalar h0lambda, h1lambda, w0lambda, w1lambda;
        for (i64 n = begin; n < end; n++) {
          for (i64 oh = 0; oh < output_height; oh++) {
            compute_source_index_and_lambda(
                ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
            for (i64 ow = 0; ow < output_width; ow++) {
              compute_source_index_and_lambda(
                  iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);

              Scalar* out = output_data + n * output_slice_size +
                  oh * output_width * channels + ow * channels;
              Scalar* i00 = input_indexr(n, ih0, iw0);
              Scalar* i01 = input_indexr(n, ih0, iw1);
              Scalar* i10 = input_indexr(n, ih1, iw0);
              Scalar* i11 = input_indexr(n, ih1, iw1);

              i64 size = channels;
              i64 d = 0;
              for (; d < size - (size % Vec::size()); d += Vec::size()) {
                Vec out_vec =
                    Vec(h0lambda * w0lambda) * Vec::loadu(i00 + d) + /* h0 * w0 * i00 */
                    Vec(h0lambda * w1lambda) * Vec::loadu(i01 + d) + /* h0 * w1 * i01 */
                    Vec(h1lambda * w0lambda) * Vec::loadu(i10 + d) + /* h1 * w0 * i10 */
                    Vec(h1lambda * w1lambda) * Vec::loadu(i11 + d);  /* h1 * w1 * i11 */
                out_vec.store(out + d);
              }
              for (; d < size; d++) {
                out[d] =
                    h0lambda * w0lambda * i00[d] + /* h0 * w0 * i00 */
                    h0lambda * w1lambda * i01[d] + /* h0 * w1 * i01 */
                    h1lambda * w0lambda * i10[d] + /* h1 * w0 * i10 */
                    h1lambda * w1lambda * i11[d];  /* h1 * w1 * i11 */
              }
            }
          }
        }
      };

      auto loop3d = [&](i64 begin, i64 end) {
        const Scalar depth_scale = area_pixel_compute_scale<Scalar>(
            input_depth, output_depth, align_corners, scales[0]);
        const Scalar height_scale = area_pixel_compute_scale<Scalar>(
            input_height, output_height, align_corners, scales[1]);
        const Scalar width_scale = area_pixel_compute_scale<Scalar>(
            input_width, output_width, align_corners, scales[2]);

        auto input_indexr = [=](i64 n, i64 d, i64 h, i64 w) {
          return input_data + n * input_depth * input_height * input_width * channels +
              d * input_height * input_width * channels +
              h * input_width * channels + w * channels;
        };

        i64 id0, id1, ih0, ih1, iw0, iw1;
        Scalar d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
        for (i64 n = begin; n < end; n++) {
          for (i64 od = 0; od < output_depth; od++) {
            compute_source_index_and_lambda(
                id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
            for (i64 oh = 0; oh < output_height; oh++) {
              compute_source_index_and_lambda(
                  ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
              for (i64 ow = 0; ow < output_width; ow++) {
                compute_source_index_and_lambda(
                    iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);

                Scalar* out = output_data + n * output_slice_size +
                    od * output_height * output_width * channels +
                    oh * output_width * channels + ow * channels;
                Scalar* i000 = input_indexr(n, id0, ih0, iw0);
                Scalar* i001 = input_indexr(n, id0, ih0, iw1);
                Scalar* i010 = input_indexr(n, id0, ih1, iw0);
                Scalar* i011 = input_indexr(n, id0, ih1, iw1);
                Scalar* i100 = input_indexr(n, id1, ih0, iw0);
                Scalar* i101 = input_indexr(n, id1, ih0, iw1);
                Scalar* i110 = input_indexr(n, id1, ih1, iw0);
                Scalar* i111 = input_indexr(n, id1, ih1, iw1);

                i64 size = channels;
                i64 d = 0;
                for (; d < size - (size % Vec::size()); d += Vec::size()) {
                  Vec out_vec =
                      Vec(d0lambda * h0lambda * w0lambda) * Vec::loadu(i000 + d) + /* d0 * h0 * w0 * i000 */
                      Vec(d0lambda * h0lambda * w1lambda) * Vec::loadu(i001 + d) + /* d0 * h0 * w1 * i001 */
                      Vec(d0lambda * h1lambda * w0lambda) * Vec::loadu(i010 + d) + /* d0 * h1 * w0 * i010 */
                      Vec(d0lambda * h1lambda * w1lambda) * Vec::loadu(i011 + d) + /* d0 * h1 * w1 * i011 */
                      Vec(d1lambda * h0lambda * w0lambda) * Vec::loadu(i100 + d) + /* d1 * h0 * w0 * i100 */
                      Vec(d1lambda * h0lambda * w1lambda) * Vec::loadu(i101 + d) + /* d1 * h0 * w1 * i101 */
                      Vec(d1lambda * h1lambda * w0lambda) * Vec::loadu(i110 + d) + /* d1 * h1 * w0 * i110 */
                      Vec(d1lambda * h1lambda * w1lambda) * Vec::loadu(i111 + d);  /* d1 * h1 * w1 * i111 */
                  out_vec.store(out + d);
                }
                for (; d < size; d++) {
                  out[d] =
                      d0lambda * h0lambda * w0lambda * i000[d] + /* d0 * h0 * w0 * i000 */
                      d0lambda * h0lambda * w1lambda * i001[d] + /* d0 * h0 * w1 * i001 */
                      d0lambda * h1lambda * w0lambda * i010[d] + /* d0 * h1 * w0 * i010 */
                      d0lambda * h1lambda * w1lambda * i011[d] + /* d0 * h1 * w1 * i011 */
                      d1lambda * h0lambda * w0lambda * i100[d] + /* d1 * h0 * w0 * i100 */
                      d1lambda * h0lambda * w1lambda * i101[d] + /* d1 * h0 * w1 * i101 */
                      d1lambda * h1lambda * w0lambda * i110[d] + /* d1 * h1 * w0 * i110 */
                      d1lambda * h1lambda * w1lambda * i111[d];  /* d1 * h1 * w1 * i111 */
                }
              }
            }
          }
        }
      };

      if (ndim == 4) {
        // upsample nearest 2d
        parallel_for(0, num_batches, internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
      } else {
        // upsample nearest 3d
        TORCH_INTERNAL_ASSERT(ndim == 5);
        parallel_for(0, num_batches, internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
      }

      if (!output_.is_contiguous(channels_last_memory_format)) {
        output_.copy_(output);
      }
        */
}

/**
  | Helper structs to use with upsample_generic_Nd_kernel_impl
  |
  */
pub struct HelperInterpBase {

}

impl HelperInterpBase {
    
    #[inline] pub fn init_indices_weights(
        output_type: ScalarType,
        output:      &mut Vec<Tensor>,
        output_size: i64,
        ndims:       i64,
        reshape_dim: i64,
        interp_size: i32)  {
        
        todo!();
        /*
            auto new_shape = vector<i64>(ndims, 1);
        new_shape[reshape_dim] = output_size;

        for (int j=0; j<interp_size; j++) {
          output.emplace_back(empty(new_shape, CPU(CppTypeToScalarType<i64>())));
          output.emplace_back(empty(new_shape, CPU(output_type)));
        }
        */
    }
}

pub struct HelperInterpNearest {
    base: HelperInterpBase,
}

impl HelperInterpNearest {

    pub const INTERP_SIZE: i32 = 1;
    
    #[inline] pub fn init_indices_weights(
        output_type: ScalarType,
        output:      &mut Vec<Tensor>,
        output_size: i64,
        ndims:       i64,
        reshape_dim: i64,
        interp_size: i32)  {
        
        todo!();
        /*
            auto new_shape = vector<i64>(ndims, 1);
        new_shape[reshape_dim] = output_size;

        for (int j=0; j<interp_size; j++) {
          output.emplace_back(empty(new_shape, CPU(CppTypeToScalarType<i64>())));
          // Defines weights for consistency, but not used
          output.emplace_back(ones(new_shape, CPU(output_type)));
        }
        */
    }

    /**
      | Compute nearest mode indices and weights for
      | each interpolated dimension
      |
      | indices_weights = {
      |      {indices_0, 1.0, },  // dim -n
      |      {indices_0, 1.0, },  // dim -(n-1)
      |      ...
      |      {indices_0, 1.0, },  // dim -1
      | }
      |
      | Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to fit input/output
      | tensors.
      |
      | Indices are already containing the strides to
      | optimize the computations
      |
      */
    #[inline] pub fn compute_indices_weights(
        scalar_type:   ScalarType,
        input_size:    i64,
        output_size:   i64,
        stride:        i64,
        ndims:         i64,
        reshape_dim:   i64,
        align_corners: bool,
        opt_scale:     Option<f64>) -> Vec<Tensor> {
        
        todo!();
        /*
            vector<Tensor> output;
        HelperInterpNearest::init_indices_weights(
          scalar_type, output, output_size, ndims, reshape_dim, HelperInterpNearest::interp_size);

        AT_DISPATCH_FLOATING_TYPES(
          scalar_type, "compute_indices_weights_nearest", [&] {

            Scalar scale = area_pixel_compute_scale<Scalar>(input_size, output_size, align_corners, opt_scale);

            auto input_index_ptr = output[0].data_ptr<i64>();
            i64 input_index;

            for (i64 i=0; i<output_size; i++) {
              const Scalar real_input_index = area_pixel_compute_source_index<Scalar>(
                  scale, i, /*align_corners=*/true, /*cubic=*/false);
              input_index = static_cast<i64>(floorf(real_input_index));
              input_index_ptr[i] = static_cast<i64>(min(input_index, input_size - 1)) * stride;
            }
          }
        );
        return output;
        */
    }
}

pub struct HelperInterpLinear {
    base: HelperInterpBase,
}

impl HelperInterpLinear {

    pub const INTERP_SIZE: i32 = 2;

    /**
      | Compute indices and weights for each
      | interpolated dimension
      |
      | indices_weights = {
      |      {indices_0, weights_0, indices_1, weights_1},  // dim -n
      |      {indices_0, weights_0, indices_1, weights_1},  // dim -(n-1)
      |      ...
      |      {indices_0, weights_0, indices_1, weights_1},  // dim -1
      | }
      |
      | Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
      | fit input/output tensors.
      |
      | Indices are already containing the strides to
      | optimize the computations
      |
      */
    #[inline] pub fn compute_indices_weights(
        scalar_type:   ScalarType,
        input_size:    i64,
        output_size:   i64,
        stride:        i64,
        ndims:         i64,
        reshape_dim:   i64,
        align_corners: bool,
        opt_scale:     Option<f64>) -> Vec<Tensor> {
        
        todo!();
        /*
            vector<Tensor> output;
        HelperInterpLinear::init_indices_weights(
          scalar_type, output, output_size, ndims, reshape_dim, HelperInterpLinear::interp_size);

        AT_DISPATCH_FLOATING_TYPES(
          scalar_type, "compute_indices_weights_linear", [&] {

            Scalar scale = area_pixel_compute_scale<Scalar>(input_size, output_size, align_corners, opt_scale);

            auto input_index0_ptr = output[0].data_ptr<i64>();
            auto lambda0_ptr = output[1].data_ptr<Scalar>();
            auto input_index1_ptr = output[2].data_ptr<i64>();
            auto lambda1_ptr = output[3].data_ptr<Scalar>();

            for (i64 i=0; i<output_size; i++) {

              compute_source_index_and_lambda<Scalar>(
                input_index0_ptr[i], input_index1_ptr[i],
                lambda0_ptr[i], lambda1_ptr[i],
                scale, i, input_size, output_size, align_corners
              );
              // put stride into indices
              // index values correspond to input indices (0, 1, 2, 3, ...)
              // when multiplied by input stride, maximum possible value
              // input_size[dim-1] * input_size[dim-2] * ... for the given dimension.
              input_index0_ptr[i] *= stride;
              input_index1_ptr[i] *= stride;
            }
          }
        );
        return output;
        */
    }
}

pub struct HelperInterpCubic {
    base: HelperInterpBase,
}

impl HelperInterpCubic {

    pub const INTERP_SIZE: i32 = 4;

    /**
      | Compute indices and weights for each
      | interpolated dimension
      |
      | indices_weights = {
      |      {indices_0, weights_0, indices_1, weights_1, ..., indices_3, weights_3},  // dim -n
      |      {indices_0, weights_0, indices_1, weights_1, ..., indices_3, weights_3},  // dim -(n-1)
      |      ...
      |      {indices_0, weights_0, indices_1, weights_1, ..., indices_3, weights_3},  // dim -1
      | }
      |
      | Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
      | fit input/output tensors.
      |
      | Indices are already containing the strides to
      | optimize the computations
      |
      */
    #[inline] pub fn compute_indices_weights(
        scalar_type:   ScalarType,
        input_size:    i64,
        output_size:   i64,
        stride:        i64,
        ndims:         i64,
        reshape_dim:   i64,
        align_corners: bool,
        opt_scale:     Option<f64>) -> Vec<Tensor> {
        
        todo!();
        /*
            vector<Tensor> output;
        HelperInterpCubic::init_indices_weights(
          scalar_type, output, output_size, ndims, reshape_dim, HelperInterpCubic::interp_size);

        AT_DISPATCH_FLOATING_TYPES(
          scalar_type, "compute_indices_weights_cubic", [&] {

            Scalar scale = area_pixel_compute_scale<Scalar>(input_size, output_size, align_corners, opt_scale);

            i64 input_index;
            i64 zero = static_cast<i64>(0);
            Scalar coeffs[4];

            i64 * idx_ptr;
            Scalar * wt_ptr;

            for (i64 i=0; i<output_size; i++) {

              const Scalar real_input_index = area_pixel_compute_source_index<Scalar>(
                  scale, i, align_corners, /*cubic=*/true);
              input_index = static_cast<i64>(floorf(real_input_index));
              get_cubic_upsample_coefficients<Scalar>(coeffs, real_input_index - input_index);

              for (int j=0; j<interp_size; j++) {
                idx_ptr = output[2 * j + 0].data_ptr<i64>();
                idx_ptr[i] = static_cast<i64>(max(min(input_index + j - 1, input_size - 1), zero)) * stride;
                wt_ptr = output[2 * j + 1].data_ptr<Scalar>();
                wt_ptr[i] = coeffs[j];
              }
            }
          }
        );
        return output;
        */
    }
}

/**
  | Generic upsampling interpolation kernel for N-d
  | case.
  |
  | Input is assumed to be like NCHW, NCL, NCKHW
  | - interpolated spatial dimension are those from
  | the end up to batch size N and number of
  | channels C.
  |
  | Internally, it uses TensorIterator to optimize
  | the computations.
  |
  | - out_ndims is the number of interpolated dims:
  | 1, 2, 3
  |
  | - scale_type is template type for scales,
  | typically optional<double>
  |
  | - template<typename> class F is one of the
  | above structs to compute indices and weights
  |
  */
pub fn upsample_generic_nd_kernel_impl<const out_ndims: i32, scale_type, F>(
    output:        &Tensor,
    input:         &Tensor,
    align_corners: bool,
    scales:        &ScaleType)  {

    todo!();
        /*
            // input can be NCHW, NCL or NCKHW
      auto shape = input.sizes().vec();
      auto strides = input.strides().vec();
      auto oshape = output.sizes();

      TORCH_INTERNAL_ASSERT(
        shape.size() == oshape.size() && shape.size() == 2 + out_ndims
      );
      TORCH_INTERNAL_ASSERT(strides.size() == 2 + out_ndims);

      for (int i=0; i<out_ndims; i++) {
        shape[i + 2] = oshape[i + 2];
        strides[i + 2] = 0;
      }
      auto restrided_input = input.as_strided(shape, strides);

      vector<vector<Tensor>> indices_weights;

      constexpr int interp_size = F::interp_size;
      auto input_scalar_type = input.scalar_type();
      if (interp_size == 1 && input_scalar_type == ScalarType::Byte) {
        // nearest also supports uint8 tensor, but we have to use float
        // with compute_indices_weights
        input_scalar_type = ScalarType::Float;
      }

      for (int i=0; i<out_ndims; i++) {
        indices_weights.emplace_back(
          F::compute_indices_weights(
            input_scalar_type, input.size(i + 2), oshape[i + 2],
            input.stride(i + 2) * input.element_size(),
            input.dim(), i + 2, align_corners, scales[i]
          )
        );
      }

      TensorIteratorConfig config;
      config.check_all_same_dtype(false)
        .declare_static_dtype_and_device(input.scalar_type(), input.device())
        .add_output(output)
        .add_input(restrided_input);

      for (auto & idx_weight: indices_weights) {
        for (auto& tensor : idx_weight) {
          config.add_input(tensor);
        }
      }

      auto iter = config.build();

      if (interp_size > 1) {
        // Nearest also supports uint8 tensor, so need to handle it separately
        AT_DISPATCH_FLOATING_TYPES(
            iter.dtype(), "upsample_generic_Nd", [&] {
            // MSVC can not catch constexpr int interp_size here
            constexpr int mode = F::interp_size;
            cpu_upsample_generic<Scalar, out_ndims, mode>(iter);
        });
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Byte,
            iter.dtype(), "upsample_generic_Nd", [&] {
            constexpr int mode = F::interp_size;
            cpu_upsample_generic<Scalar, out_ndims, mode>(iter);
        });
      }
        */
}

pub fn upsample_nearest1d_kernel_impl(
        output:   &Tensor,
        input:    &Tensor,
        scales_w: Option<f64>)  {
    
    todo!();
        /*
            upsample_generic_Nd_kernel_impl<1, scale_t, HelperInterpNearest>(
        output, input, false, {scales_w});
        */
}

pub fn upsample_nearest2d_kernel_impl(
        output:   &Tensor,
        input:    &Tensor,
        scales_h: Option<f64>,
        scales_w: Option<f64>)  {
    
    todo!();
        /*
            if (input.is_contiguous(MemoryFormat::ChannelsLast)) {
        AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Byte, input.scalar_type(), "upsample_nearest2d_channels_last", [&] {
          cpu_upsample_nearest_channels_last<Scalar, scale_t>(output, input, {scales_h, scales_w});
        });
      } else {
        upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpNearest>(
          output, input, false, {scales_h, scales_w});
      }
        */
}

pub fn upsample_nearest3d_kernel_impl(
        output:   &Tensor,
        input:    &Tensor,
        scales_d: Option<f64>,
        scales_h: Option<f64>,
        scales_w: Option<f64>)  {
    
    todo!();
        /*
            if (input.is_contiguous(MemoryFormat::ChannelsLast3d)) {
        AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Byte, input.scalar_type(), "upsample_nearest3d_channels_last", [&] {
          cpu_upsample_nearest_channels_last<Scalar, scale_t>(output, input, {scales_d, scales_h, scales_w});
        });
      } else {
        upsample_generic_Nd_kernel_impl<3, scale_t, HelperInterpNearest>(
          output, input, false, {scales_d, scales_h, scales_w});
      }
        */
}

pub fn upsample_linear1d_kernel_impl(
        output:        &Tensor,
        input:         &Tensor,
        align_corners: bool,
        scales_w:      Option<f64>)  {
    
    todo!();
        /*
            upsample_generic_Nd_kernel_impl<1, scale_t, HelperInterpLinear>(
        output, input, align_corners, {scales_w});
        */
}

pub fn upsample_bilinear2d_kernel_impl(
        output:        &Tensor,
        input:         &Tensor,
        align_corners: bool,
        scales_h:      Option<f64>,
        scales_w:      Option<f64>)  {
    
    todo!();
        /*
            // Temporarily dispatch to original channels last implementation
      if (input.is_contiguous(MemoryFormat::ChannelsLast)) {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_bilinear2d_channels_last", [&] {
          cpu_upsample_linear_channels_last<Scalar, scale_t>(output, input, align_corners, {scales_h, scales_w});
        });
      } else {
        upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpLinear>(
          output, input, align_corners, {scales_h, scales_w});
      }
        */
}

pub fn upsample_trilinear3d_kernel_impl(
        output:        &Tensor,
        input:         &Tensor,
        align_corners: bool,
        scales_d:      Option<f64>,
        scales_h:      Option<f64>,
        scales_w:      Option<f64>)  {
    
    todo!();
        /*
            if (input.is_contiguous(MemoryFormat::ChannelsLast3d)) {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_trilinear3d_channels_last", [&] {
          cpu_upsample_linear_channels_last<Scalar, scale_t>(output, input, align_corners, {scales_d, scales_h, scales_w});
        });
      } else {
        upsample_generic_Nd_kernel_impl<3, scale_t, HelperInterpLinear>(
          output, input, align_corners, {scales_d, scales_h, scales_w});
      }
        */
}

pub fn upsample_bicubic2d_kernel_impl(
        output:        &Tensor,
        input:         &Tensor,
        align_corners: bool,
        scales_h:      Option<f64>,
        scales_w:      Option<f64>)  {
    
    todo!();
        /*
            upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpCubic>(
        output, input, align_corners, {scales_h, scales_w});
        */
}

pub fn cpu_upsample_nearest_backward<Scalar, scale_type>(
        grad_input:  &Tensor,
        grad_output: &Tensor,
        scales:      &ScaleType)  {

    todo!();
        /*
            TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
                  " for `grad_input` but got dtype ", grad_input_.dtype());

      auto grad_output = grad_output_.contiguous();
      auto grad_input = grad_input_.contiguous();

      auto grad_output_data = grad_output.data_ptr<Scalar>();
      auto grad_input_data = grad_input.data_ptr<Scalar>();
      auto input_sizes = grad_input.sizes().vec();
      auto output_sizes = grad_output.sizes().vec();
      auto ndim = input_sizes.size();

      // treat nbatch and channels as one dimension
      i64 channels = input_sizes[0] * input_sizes[1];
      i64 input_depth = (ndim == 5) ? input_sizes[2] : 1;
      i64 output_depth = (ndim == 5) ? output_sizes[2] : 1;
      i64 input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
      i64 output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
      i64 input_width = input_sizes[ndim - 1];
      i64 output_width = output_sizes[ndim - 1];

      i64 output_slice_size = output_depth * output_height * output_width;
      i64 input_slice_size = input_depth * input_height * input_width;

      auto loop1d = [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++){
          for (i64 ow = 0; ow < output_width; ow++) {
            i64 iw = nearest_idx(ow, input_width, output_width, scales[0]);
            i64 output_offset = c * output_slice_size + ow;
            i64 input_offset = c * input_slice_size + iw;
            grad_input_data[input_offset] += grad_output_data[output_offset];
          }
        }
      };

      auto loop2d = [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++) {
          for (i64 oh = 0; oh < output_height; oh++) {
            i64 ih = nearest_idx(oh, input_height, output_height, scales[0]);
            for (i64 ow = 0; ow < output_width; ow++) {
              i64 iw = nearest_idx(ow, input_width, output_width, scales[1]);
              i64 output_offset = c * output_slice_size + oh * output_width + ow;
              i64 input_offset = c * input_slice_size + ih * input_width + iw;
              grad_input_data[input_offset] += grad_output_data[output_offset];
            }
          }
        }
      };

      auto loop3d = [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++) {
          for (i64 od = 0; od < output_depth; od++) {
            i64 id = nearest_idx(od, input_depth, output_depth, scales[0]);
            for (i64 oh = 0; oh < output_height; oh++) {
              i64 ih = nearest_idx(oh, input_height, output_height, scales[1]);
              for (i64 ow = 0; ow < output_width; ow++) {
                i64 iw = nearest_idx(ow, input_width, output_width, scales[2]);
                i64 output_offset = c * output_slice_size +
                    od *  output_height * output_width + oh * output_width + ow;
                i64 input_offset = c * input_slice_size +
                    id * input_height * input_width + ih * input_width + iw;
                grad_input_data[input_offset] += grad_output_data[output_offset];
              }
            }
          }
        }
      };

      if (ndim == 3) {
        // upsample nearest 1d
        parallel_for(0, channels, internal::GRAIN_SIZE / output_slice_size, loop1d);
      } else if (ndim == 4) {
        // upsample nearest 2d
        parallel_for(0, channels, internal::GRAIN_SIZE / output_slice_size , loop2d);
      } else {
        // upsample nearest 3d
        TORCH_INTERNAL_ASSERT(ndim == 5);
        parallel_for(0, channels, internal::GRAIN_SIZE / output_slice_size, loop3d);
      }

      if (!grad_input_.is_contiguous()) {
        grad_input_.copy_(grad_input);
      }
        */
}

pub fn upsample_nearest1d_backward_kernel_impl(
        grad_input:  &Tensor,
        grad_output: &Tensor,
        scales_w:    Option<f64>)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_nearest1d_backward", [&] {
        cpu_upsample_nearest_backward<Scalar, scale_t>(grad_input, grad_output, {scales_w});
      });
        */
}

pub fn upsample_nearest2d_backward_kernel_impl(
        grad_input:  &Tensor,
        grad_output: &Tensor,
        scales_h:    Option<f64>,
        scales_w:    Option<f64>)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_nearest2d_backward", [&] {
        cpu_upsample_nearest_backward<Scalar, scale_t>(grad_input, grad_output, {scales_h, scales_w});
      });
        */
}

pub fn upsample_nearest3d_backward_kernel_impl(
        grad_input:  &Tensor,
        grad_output: &Tensor,
        scales_d:    Option<f64>,
        scales_h:    Option<f64>,
        scales_w:    Option<f64>)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_nearest3d_backward", [&] {
        cpu_upsample_nearest_backward<Scalar, scale_t>(grad_input, grad_output, {scales_d, scales_h, scales_w});
      });
        */
}

register_dispatch!{upsample_nearest1d_kernel          , &upsample_nearest1d_kernel_impl}
register_dispatch!{upsample_nearest2d_kernel          , &upsample_nearest2d_kernel_impl}
register_dispatch!{upsample_nearest3d_kernel          , &upsample_nearest3d_kernel_impl}
register_dispatch!{upsample_nearest1d_backward_kernel , &upsample_nearest1d_backward_kernel_impl}
register_dispatch!{upsample_nearest2d_backward_kernel , &upsample_nearest2d_backward_kernel_impl}
register_dispatch!{upsample_nearest3d_backward_kernel , &upsample_nearest3d_backward_kernel_impl}
register_dispatch!{upsample_linear1d_kernel           , &upsample_linear1d_kernel_impl}
register_dispatch!{upsample_bilinear2d_kernel         , &upsample_bilinear2d_kernel_impl}
register_dispatch!{upsample_trilinear3d_kernel        , &upsample_trilinear3d_kernel_impl}
register_dispatch!{upsample_bicubic2d_kernel          , &upsample_bicubic2d_kernel_impl}
