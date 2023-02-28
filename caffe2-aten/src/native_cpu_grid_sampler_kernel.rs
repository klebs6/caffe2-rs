/*!  
 |   NOTE [ Grid Sample CPU Kernels ]
 |
 |   Implementation of vectorized grid sample CPU kernels is divided into three
 |   parts. More detailed description exist after this paragraph, but on a high
 |   level, they are
 |
 |   1. `ComputeLocation` struct
 |      + Computes the interpolation location basing on padding mode.
 |
 |   2. `ApplyGridSample` struct
 |      + Owns N (# spatial dims) `ComputeLocation` structs, and uses them to
 |        compute the interpolation locations.
 |      + Interpolates the values and writes to output.
 |
 |   3. `grid_sample_2d_grid_slice_iterator` function
 |
 |      + Iterates over a slice of the grid tensor based on the geometry by the
 |        spatial ordering, i.e., the first iteration will process grid values
 |           grid[n, 0, 0, :], grid[n, 0, 1, :], grid[n, 0, 2, :], ...
 |        (Recall that, e.g., 2D grid has shape [N x H x W x 2], so grid[n, ...]
 |         is a slice, and grid[n, h, w, :] contains the values for a single
 |         output spatial location.)
 |
 |      + Applies a given operator at each iteration, so we can use the same
 |        pattern for forward and backward.
 |
 |   Putting everything together, we have, e.g., the forward kernel implemented
 |   as
 |
 |      // `ApplyGridSample` struct that processes grid values, extracts and
 |      // interpolates input values, and write to output.
 |      ApplyGridSample<Scalar, 2, interp, padding> grid_sample(input_accessor);
 |
 |      // For each slice, we call `grid_sample_2d_grid_slice_iterator` with
 |      //   1. the grid slice, and
 |      //   2. a lambda that takes in
 |      //      i.   location vectors (x and y for 2D) extracted from grid
 |      //      ii.  `spatial_offset` as the spatial offset of these vectors
 |      //           from the beginning of this slice.
 |      //      iii. `len` as the number of valid locations in the vectors.
 |      //           (There might not be enough near boundary.)
 |      for (int n = 0; n < input_accessor.size(0); n++) {
 |        grid_sample_2d_grid_slice_iterator(
 |          grid_accessor[n],
 |          [&](const Vectorized<Scalar>& grid_x,
 |              const Vectorized<Scalar>& grid_y,
 |              i64 spatial_offset, i64 len) {
 |            grid_sample.forward(out_accessor[n], input_accessor[n],
 |                                spatial_offset, grid_x, grid_y, len);
 |          });
 |      }
 |
 |   Now we talk about details of each of these three parts:
 |
 |   1. `ComputeLocation` struct
 |      Transforms grid values into interpolation locations of the input tensor
 |      for a particular spatial dimension, based on the size of that dimension
 |      in input tensor, and the padding mode.
 |
 |        template<typename Scalar, GridSamplerPadding padding>
 |        struct ComputeLocation {
 |          using Vec = Vectorized<Scalar>;
 |
 |          // ctor
 |          ComputeLocation(i64 size);
 |
 |          // Given grid values `in`, return the interpolation locations after
 |          // un-normalization and padding mechanism (elementwise).
 |          Vec apply(const Vec &in) const;
 |
 |          // Similar to `apply`, but also returns `d apply(in) / d in`
 |          // (elementwise).
 |          // this is often used in gradient computation.
 |          pair<Vec, Vec> apply_get_grad(const Vec &in) const;
 |        };
 |
 |   2. `ApplyGridSample` struct
 |      Owns N `ComputeLocation` structs, where N is the number of spatial
 |      dimensions. Given N input grid vectors (one for each spatial dimension)
 |      and spatial offset, it gets the interpolation locations from
 |      `ComputeLocation`s, applies interpolation procedure, and then writes to
 |      the output (or grad_input & grad_grid in backward).
 |
 |        template<typename Scalar, int spatial_dim,
 |                 GridSamplerInterpolation interp,
 |                 GridSamplerPadding padding>
 |        struct ApplyGridSample {
 |
 |          // ctor
 |          ApplyGridSample(const TensorAccessor<Scalar, 4>& input);
 |
 |          // Applies grid sampling (forward) procedure:
 |          //   1. computes interpolation locations from grid values `grid_x`
 |          //      and `grid_y`,
 |          //   2. interpolates output values using the locations and input
 |          //      data in `inp_slice`, and
 |          //   3. writes the first `len` values in the interpolated vector to
 |          //      `out_slice` with spatial offset being `offset`.
 |          //
 |          // This assimes that `grid_x` and `grid_y` all contain valid grid
 |          // values \in [-1, 1], even at indices greater than `len`.
 |          //
 |          // The `*_slice` argument namess mean samples within a batch (i.e.,
 |          // with the batch dimension sliced out).
 |          void forward(TensorAccessor<Scalar, 3>& out_slice,
 |                       const TensorAccessor<Scalar, 3>& inp_slice,
 |                       i64 offset, const Vec& grid_x, const Vec& grid_y,
 |                       i64 len) const;
 |
 |          // Applies grid sampling (backward) procedure. Arguments semantics
 |          // and strategy are similar to those of `forward`.
 |          void backward(TensorAccessor<Scalar, 3>& gInp_slice,
 |                        TensorAccessor<Scalar, 3>& gGrid_slice,
 |                        const TensorAccessor<Scalar, 3>& gOut_slice,
 |                        const TensorAccessor<Scalar, 3>& inp_slice,
 |                        i64 offset, const Vec& grid_x, const Vec& grid_y,
 |                        i64 len) const;
 |        };
 |
 |   3. `grid_sample_2d_grid_slice_iterator` function
 |      Among the tensors we work with, we know that the output tensors are
 |      contiguous (i.e., `output` in forward, and `grad_input` & `grad_grid` in
 |      backward), we need to randomly read `input` anyways, and `grad_output`
 |      usually comes from autograd and is often contiguous. So we base our
 |      iterating strategy on the geometry of grid.
 |      `grid_sample_2d_grid_slice_iterator` function provides an abstraction to
 |      efficiently iterates through a `grid` slice (without batch dimension).
 |      See comments of that function on the specific cases and strategies used.
 |
 |        template<typename Scalar, typename ApplyFn>
 |        void grid_sample_2d_grid_slice_iterator(
 |          const TensorAccessor<Scalar, 3>& grid_slice,
 |          const ApplyFn &apply_fn);
 |
 |      `apply_fn` is a function/lambda that takes in
 |           i.   location vectors (x and y for 2D) extracted from grid
 |           ii.  `spatial_offset` as the spatial offset of these vectors
 |                from the beginning of this slice.
 |           iii. `len` as the number of valid locations in the vectors.
 |                (There might not be enough near boundary.)

 |       It should be callable as if it has declaration:
 |          void apply_fn(const Vectorized<Scalar>& grid_x,
 |                        const Vectorized<Scalar>& grid_y,
 |                        i64 spatial_offset, i64 len);
 |
 |      `apply_fn` will be called multiple times, and together cover the entire
 |      output spatial space.
 |
 |  Now you should be able tp understand everything about the implementation of
 |  2D forward kernel shown at the beginning of this note.
 |
 **/

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/GridSamplerKernel.h]

lazy_static!{
    /*
    using forward_2d_fn = Tensor(*)(const Tensor &, const Tensor &, i64, i64, bool);
    using backward_2d_fn = tuple<Tensor, Tensor>(*)(const Tensor &, const Tensor &, const Tensor &, i64, i64, bool);
    */
}

declare_dispatch!{forward_2d_fn, grid_sampler_2d_cpu_kernel}
declare_dispatch!{backward_2d_fn, grid_sampler_2d_backward_cpu_kernel}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/GridSamplerKernel.cpp]

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ComputeLocation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ComputeLocationBaseWithAlignedCorners<Scalar> {

    /**
      | values are clipped to between 0 and max_val
      |
      */
    max_val:        Scalar,

    /**
      | unnormalization scaling factor
      |
      */
    scaling_factor: Scalar,

    /**
      | reflection parameters: reflected
      | coordinates land in [low, low+span]
      | inclusive only used when align_corners=False
      |
      */
    low:            Scalar,

    twice_span:     Scalar,

    /**
      | if the reflecting span is empty, all
      | reflected coords are set to 0
      |
      */
    empty:          bool,
}

pub struct ComputeLocationBaseWithUnalignedCorners<Scalar> {

    /**
      | values are clipped to between 0 and max_val
      |
      */
    max_val:        Scalar,

    /**
      | unnormalization scaling factor
      |
      */
    scaling_factor: Scalar,

    /**
      | reflection parameters: reflected
      | coordinates land in [low, low+span]
      | inclusive
      |
      */
    low:            Scalar,

    twice_span:     Scalar,

    /**
      | if the reflecting span is empty, all
      | reflected coords are set to 0 only used
      | when align_corners=True
      |
      */
    empty:          bool,
}

/**
  | Struct to compute interpolation location from
  | grid values, and to apply padding mechanism
  | (e.g., reflection).
  |
  | See NOTE [ Grid Sample CPU Kernels ] for
  | details.
  */
pub enum ComputeLocationBase<Scalar> {
    AlignCorners(ComputeLocationBaseWithAlignedCorners<Scalar>),
    UnalignedCorners(ComputeLocationBaseWithUnalignedCorners<Scalar>),
}

impl<Scalar> ComputeLocationBase<Scalar> {
    
    pub fn new_aligned_corners(size: i64) -> Self {
    
        todo!();
        /*


            : max_val(static_cast<Scalar>(size - 1))
        , scaling_factor(static_cast<Scalar>(size - 1) / 2)
        , low(static_cast<Scalar>(0))
        , twice_span(static_cast<Scalar>(size - 1) * 2)
        , empty(size <= 1)
        */
    }
    
    #[inline] pub fn aligned_corners_unnormalize(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            return (in + Vec(1)) * Vec(scaling_factor);
        */
    }
    
    #[inline] pub fn aligned_corners_clip_coordinates(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            // Invert order of clamp_min operands in order to clamp Nans to zero
        return clamp_max(Vec(max_val), clamp_min(Vec(0), in));
        */
    }

    /**
      | same as clip_coordinates but also returns
      | the gradient multiplier
      |
      */
    #[inline] pub fn aligned_corners_clip_coordinates_get_grad(&self, in_: &Vec) -> (Vec,Vec) {
        
        todo!();
        /*
            using int_t = int_same_size_t<Scalar>;
        auto bounded_lo = maximum(in, Vec(0));
        // Integral type equality comparison is very very fast because it just looks
        // at the bits. Casting is free too. So we use the following pattern instead
        // of comparison + blendv.
        // Note that it is important for the gradient calculation that borders
        // are considered out of bounds.
        auto in_bound_lo = cast<Scalar>(cast<int_t>(bounded_lo) != cast<int_t>(Vec(0)));
        auto res = minimum(bounded_lo, Vec(max_val));
        auto in_bound_hi = cast<Scalar>(cast<int_t>(res) != cast<int_t>(Vec(max_val)));
        return make_pair(res, in_bound_lo & in_bound_hi);
        */
    }
    
    #[inline] pub fn aligned_corners_reflect_coordinates(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            if (empty) {
          return Vec(0);
        }
        Vec twice_span_vec(twice_span);
        auto abs_in = in.abs();
        auto fdouble_flips = abs_in / twice_span_vec;
        auto double_flips = fdouble_flips.trunc();
        auto extra = abs_in - double_flips * twice_span_vec;
        // Now we need to test if extra > max_val to find out if another flip is
        // needed. The following comparison does that and returns the correct
        // flipped value.
        return minimum(extra, twice_span_vec - extra);
        */
    }

    /**
      | same as reflect_coordinates but also
      | returns the gradient multiplier
      |
      */
    #[inline] pub fn aligned_corners_reflect_coordinates_get_grad(&self, in_: &Vec) -> (Vec,Vec) {
        
        todo!();
        /*
            if (empty) {
          return make_pair(Vec(0), Vec(0));
        }
        Vec twice_span_vec(twice_span);
        auto neg_in = in < Vec(0);
        auto abs_in = in.abs();
        auto fdouble_flips = abs_in / twice_span_vec;
        auto double_flips = fdouble_flips.trunc();

        auto extra = abs_in - double_flips * twice_span_vec;
        auto reflected_extra = twice_span_vec - extra;
        auto one_more_flip = extra > reflected_extra;

        return make_pair(
          Vec::blendv(extra, reflected_extra, one_more_flip),
          Vec::blendv(Vec(1), Vec(-1), one_more_flip ^ neg_in)
        );
        */
    }
    
    pub fn new_unaligned_corners(size: i64) -> Self {
    
        todo!();
        /*


            : max_val(static_cast<Scalar>(size - 1))
        , scaling_factor(static_cast<Scalar>(size) / 2)
        , low(static_cast<Scalar>(-0.5))
        , twice_span(static_cast<Scalar>(size) * 2)
        , empty(size <= 0)
        */
    }
    
    #[inline] pub fn unaligned_corners_unnormalize(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            return (in + Vec(1)) * Vec(scaling_factor) - Vec(0.5);
        */
    }
    
    #[inline] pub fn unaligned_corners_clip_coordinates(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            // Invert order of clamp_min operands in order to clamp Nans to zero
        return clamp_max(Vec(max_val), clamp_min(Vec(0), in));
        */
    }

    /**
      | same as clip_coordinates but also returns
      | the gradient multiplier
      |
      */
    #[inline] pub fn unaligned_corners_clip_coordinates_get_grad(&self, in_: &Vec) -> (Vec,Vec) {
        
        todo!();
        /*
            using int_t = int_same_size_t<Scalar>;
        auto bounded_lo = maximum(in, Vec(0));
        // Integral type equality comparison is very very fast because it just looks
        // at the bits. Casting is free too. So we use the following pattern instead
        // of comparison + blendv.
        // Note that it is important for the gradient calculation that borders
        // are considered out of bounds.
        auto in_bound_lo = cast<Scalar>(cast<int_t>(bounded_lo) != cast<int_t>(Vec(0)));
        auto res = minimum(bounded_lo, Vec(max_val));
        auto in_bound_hi = cast<Scalar>(cast<int_t>(res) != cast<int_t>(Vec(max_val)));
        return make_pair(res, in_bound_lo & in_bound_hi);
        */
    }
    
    #[inline] pub fn unaligned_corners_reflect_coordinates(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            Vec twice_span_vec(twice_span), low_vec(low);
        // Since reflection is around low and low+span, subtract low before
        // the reflection, and then add it back at the end.
        auto abs_in = (in - low_vec).abs();
        auto fdouble_flips = abs_in / twice_span_vec;
        auto double_flips = fdouble_flips.trunc();
        auto extra = abs_in - double_flips * twice_span_vec;
        // Now we need to test if extra > max_val to find out if another flip is
        // needed. The following comparison does that and returns the correct
        // flipped value.
        return minimum(extra, twice_span_vec - extra) + low_vec;
        */
    }

    /**
      | same as reflect_coordinates but also
      | returns the gradient multiplier
      |
      */
    #[inline] pub fn unaligned_corners_reflect_coordinates_get_grad(&self, in_: &Vec) -> (Vec,Vec) {
        
        todo!();
        /*
            Vec twice_span_vec(twice_span), low_vec(low);
        Vec in_minus_low = in - low_vec;
        auto neg_in = in_minus_low < Vec(0);
        auto abs_in = in_minus_low.abs();
        auto fdouble_flips = abs_in / twice_span_vec;
        auto double_flips = fdouble_flips.trunc();

        auto extra = abs_in - double_flips * twice_span_vec;
        auto reflected_extra = twice_span_vec - extra;
        auto one_more_flip = extra > reflected_extra;

        return make_pair(
          Vec::blendv(extra, reflected_extra, one_more_flip) + low_vec,
          Vec::blendv(Vec(1), Vec(-1), one_more_flip ^ neg_in)
        );
        */
    }
}

/**
  | template<typename Scalar, GridSamplerPadding
  | padding, bool align_corners> struct ComputeLocation;
  |
  */
pub struct ComputeLocationZeroesPaddingWithAlignedCorners<Scalar> {
    base: ComputeLocationBaseWithAlignedCorners<Scalar>,
}

pub mod compute_location_zeroes_padding_with_aligned_corners {

    use super::*;

    lazy_static!{
        /*
        using Vec = Vectorized<Scalar>;
        using ComputeLocationBase<Scalar, align_corners>::unnormalize;
        using ComputeLocationBase<Scalar, align_corners>::scaling_factor;
        using ComputeLocationBase<Scalar, align_corners>::ComputeLocationBase;
        */
    }
}

impl<Scalar> ComputeLocationZeroesPaddingWithAlignedCorners<Scalar> {

    #[inline] pub fn apply(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            return unnormalize(in);
        */
    }
    
    #[inline] pub fn compute_coordinates(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            return in;
        */
    }
    
    #[inline] pub fn apply_get_grad(&self, in_: &Vec) -> (Vec,Vec) {
        
        todo!();
        /*
            return make_pair(unnormalize(in), Vec(scaling_factor));
        */
    }
}

pub struct ComputeLocationBorderPaddingWithAlignedCorners<Scalar> {
    base: ComputeLocationBaseWithAlignedCorners<Scalar>,
}

pub mod compute_location_border_padding_with_aligned_corners {

    use super::*;

    lazy_static!{
        /*
          using Vec = Vectorized<Scalar>;
          using ComputeLocationBase<Scalar, align_corners>::unnormalize;
          using ComputeLocationBase<Scalar, align_corners>::clip_coordinates;
          using ComputeLocationBase<Scalar, align_corners>::clip_coordinates_get_grad;
          using ComputeLocationBase<Scalar, align_corners>::scaling_factor;
          using ComputeLocationBase<Scalar, align_corners>::ComputeLocationBase;
        */
    }
}

impl<Scalar> ComputeLocationBorderPaddingWithAlignedCorners<Scalar> {
    
    #[inline] pub fn apply(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            return clip_coordinates(unnormalize(in));
        */
    }
    
    #[inline] pub fn compute_coordinates(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            return clip_coordinates(in);
        */
    }
    
    #[inline] pub fn apply_get_grad(&self, in_: &Vec) -> (Vec,Vec) {
        
        todo!();
        /*
            Vec res, grad_clip;
        tie(res, grad_clip) = clip_coordinates_get_grad(unnormalize(in));
        return make_pair(res, grad_clip & Vec(scaling_factor));
        */
    }
}

pub struct ComputeLocationReflectionPaddingWithAlignedCorners<Scalar> {
    base: ComputeLocationBaseWithAlignedCorners<Scalar>,
}

pub mod compute_location_reflection_padding_with_aligned_corners {

    use super::*;

    lazy_static!{
        /*
        using Vec = Vectorized<Scalar>;
          using ComputeLocationBase<Scalar, align_corners>::unnormalize;
          using ComputeLocationBase<Scalar, align_corners>::clip_coordinates;
          using ComputeLocationBase<Scalar, align_corners>::clip_coordinates_get_grad;
          using ComputeLocationBase<Scalar, align_corners>::reflect_coordinates;
          using ComputeLocationBase<Scalar, align_corners>::reflect_coordinates_get_grad;
          using ComputeLocationBase<Scalar, align_corners>::scaling_factor;
          using ComputeLocationBase<Scalar, align_corners>::ComputeLocationBase;
        */
    }
}

impl<Scalar> ComputeLocationReflectionPaddingWithAlignedCorners<Scalar> {
    
    #[inline] pub fn apply(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            auto res = reflect_coordinates(unnormalize(in));
        res = clip_coordinates(res);
        return res;
        */
    }
    
    #[inline] pub fn compute_coordinates(&self, in_: &Vec) -> Vec {
        
        todo!();
        /*
            auto res = reflect_coordinates(in);
        res = clip_coordinates(res);
        return res;
        */
    }
    
    #[inline] pub fn apply_get_grad(&self, in_: &Vec) -> (Vec,Vec) {
        
        todo!();
        /*
            Vec res, grad_refl, grad_clip, grad(scaling_factor);
        tie(res, grad_refl) = reflect_coordinates_get_grad(unnormalize(in));
        grad = grad_refl * grad;
        tie(res, grad_clip) = clip_coordinates_get_grad(res);
        grad = grad_clip & grad;
        return make_pair(res, grad);
        */
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ApplyGridSample ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
  | Struct to apply grid sample (reading from
  | input, interpolate, and write to output).
  |
  | See NOTE [ Grid Sample CPU Kernels ] for
  | details.
  */
#[inline] pub fn mask_scatter_add<Scalar>(
        src:       *const Scalar,
        base_addr: *mut Scalar,
        offsets:   *const IntSameSize<Scalar>,
        mask:      *const IntSameSize<Scalar>,
        len:       i64)  {

    todo!();
        /*
            #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (i64 i = 0; i < len; i++) {
        if (mask[i] & 0x01) {
          base_addr[offsets[i]] += src[i];
        }
      }
        */
}

pub struct InterpParams<Scalar> {
    distances_to_4sides:                [Vec<Scalar>; 4],
    interpolation_weights_wrt_4corners: [Vec<Scalar>; 4],
    inbound_masks:                      [Vec<Scalar>; 4],
    y_n:                                Vec<IntSameSize<Scalar>>,
    x_w:                                Vec<IntSameSize<Scalar>>,
}

/**
  | template<typename Scalar, 
  | int spatial_dim,
  | GridSamplerInterpolation interp,
  | GridSamplerPadding padding, 
  | bool align_corners> struct ApplyGridSample;
  |
  */
pub enum ApplyGridSample<Scalar, const Padding: GridSamplerPadding, const AlignCorners: bool> {

    /// SpatialDim == 2
    /// GridSamplerInterpolation::Bilinear
    Bilinear2 {
        inp_h:     i64,
        inp_w:     i64,
        inp_sh:    i64,
        inp_sw:    i64,
        c:         i64,
        inp_sc:    i64,
        compute_h: ComputeLocation<Scalar,Padding,AlignCorners>,
        compute_w: ComputeLocation<Scalar,Padding,AlignCorners>,

        /**
          | default == (padding != GridSamplerPadding::Zeros);
          |
          */
        must_in_bound: bool,
    },

    /// SpatialDim == 2
    /// GridSamplerInterpolation::Nearest
    Nearest2 {
        inp_h:         i64,
        inp_w:         i64,
        inp_sh:        i64,
        inp_sw:        i64,
        c:             i64,
        inp_sc:        i64,
        compute_h:     ComputeLocation<Scalar,Padding,AlignCorners>,
        compute_w:     ComputeLocation<Scalar,Padding,AlignCorners>,

        /**
          | == (padding != GridSamplerPadding::Zeros);
          |
          */
        must_in_bound: bool,
    },

    /// SpatialDim == 2
    /// GridSamplerInterpolation::Bicubic
    ///
    /// Use bicubic convolution algorithm. Based on
    /// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    Bicubic2 {
        inp_h:         i64,
        inp_w:         i64,
        inp_sh:        i64,
        inp_sw:        i64,
        c:             i64,
        inp_sc:        i64,
        compute_h:     ComputeLocation<Scalar,Padding,AlignCorners>,
        compute_w:     ComputeLocation<Scalar,Padding,AlignCorners>,

        /**
          | == (padding != GridSamplerPadding::Zeros);
          |
          */
        must_in_bound: bool,

        /**
          | constant used in cubic convolution
          | could be -0.5 or -0.75, use the same value
          | in UpSampleBicubic2d.h default = Vec(-0.75);
          |
          */
        a:             Vec,
    }
}

impl<Scalar, const Padding: GridSamplerPadding, const AlignCorners: bool> 
ApplyGridSample<Scalar, Padding, AlignCorners> 
{
    pub fn new_bilinear2(input: &TensorAccessor<Scalar,4>) -> Self {
    
        todo!();
        /*


            : inp_H(input.size(2))
        , inp_W(input.size(3))
        , inp_sH(input.stride(2))
        , inp_sW(input.stride(3))
        , C(input.size(1))
        , inp_sC(input.stride(1))
        , compute_H(input.size(2))
        , compute_W(input.size(3))
        */
    }

    #[inline] pub fn bilinear2_compute_interp_params(&self, x: &Vec, y: &Vec) -> InterpParams<Scalar> {
        
        todo!();
        /*
            // get NE, NW, SE, SW pixel values from (x, y)
        // assuming we get exact integer representation and just use Scalar
        // if we don't, the weights will be garbage anyways.
        auto x_w = x.floor();
        auto y_n = y.floor();

        // get distances to each side
        auto w = x - x_w;
        auto e = Vec(1) - w;
        auto n = y - y_n;
        auto s = Vec(1) - n;

        // get interpolation weights for each neighbor
        // e.g., for the nw corder, the weight is `dist_to_south * dist_to_east`.
        auto nw = s * e;
        auto ne = s * w;
        auto sw = n * e;
        auto se = n * w;

        auto i_x_w = convert_to_int_of_same_size(x_w);
        auto i_y_n = convert_to_int_of_same_size(y_n);
        auto i_x_e = i_x_w + iVec(1);
        auto i_y_s = i_y_n + iVec(1);

        // Use int comparison because it is much faster than float comp with AVX2
        // (latency 1 cyc vs. 4 cyc on skylake)
        // Avoid using the le and ge because those are not implemented in AVX2 and
        // are actually simulated using multiple instructions.
        auto w_mask = must_in_bound ? iVec(-1)  // true = all ones
                                    : (i_x_w > iVec(-1)) & (i_x_w < iVec(inp_W));
        auto n_mask = must_in_bound ? iVec(-1)  // true = all ones
                                    : (i_y_n > iVec(-1)) & (i_y_n < iVec(inp_H));
        auto e_mask = must_in_bound ? (i_x_e < iVec(inp_W))
                                    : (i_x_e > iVec(-1)) & (i_x_e < iVec(inp_W));
        auto s_mask = must_in_bound ? (i_y_s < iVec(inp_H))
                                    : (i_y_s > iVec(-1)) & (i_y_s < iVec(inp_H));
        auto nw_mask = cast<Scalar>(must_in_bound ? iVec(-1) : (w_mask & n_mask));
        auto ne_mask = cast<Scalar>(e_mask & n_mask);
        auto sw_mask = cast<Scalar>(w_mask & s_mask);
        auto se_mask = cast<Scalar>(e_mask & s_mask);

        return make_tuple(
          n, s, w, e,
          nw, ne, sw, se,
          nw_mask, ne_mask, sw_mask, se_mask,
          i_y_n, i_x_w);
        */
    }
    
    #[inline] pub fn bilinear2_forward(&self, 
        out_slice: &mut TensorAccessor<Scalar,3>,
        inp_slice: &TensorAccessor<Scalar,3>,
        offset:    i64,
        grid_x:    &Vec,
        grid_y:    &Vec,
        len:       i64)  {
        
        todo!();
        /*
            auto x = compute_W.apply(grid_x);
        auto y = compute_H.apply(grid_y);

        auto interp_params = compute_interp_params(x, y);

        auto nw = get<4>(interp_params);
        auto ne = get<5>(interp_params);
        auto sw = get<6>(interp_params);
        auto se = get<7>(interp_params);

        auto nw_mask = get<8>(interp_params);
        auto ne_mask = get<9>(interp_params);
        auto sw_mask = get<10>(interp_params);
        auto se_mask = get<11>(interp_params);

        auto i_y_n = get<12>(interp_params);
        auto i_x_w = get<13>(interp_params);

        auto i_nw_offset = i_y_n * iVec(inp_sH) + i_x_w * iVec(inp_sW);
        auto i_ne_offset = i_nw_offset + iVec(inp_sW);
        auto i_sw_offset = i_nw_offset + iVec(inp_sH);
        auto i_se_offset = i_sw_offset + iVec(inp_sW);

        #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (i64 c = 0; c < C; ++c) {
          auto inp_slice_C_ptr = inp_slice[c].data();

          // mask_gather zeros out the mask, so we need to make copies
          Vec nw_mask_copy = nw_mask;
          Vec ne_mask_copy = ne_mask;
          Vec sw_mask_copy = sw_mask;
          Vec se_mask_copy = se_mask;
          auto nw_val = mask_gather<sizeof(Scalar)>(Vec(0), inp_slice_C_ptr, i_nw_offset, nw_mask_copy);
          auto ne_val = mask_gather<sizeof(Scalar)>(Vec(0), inp_slice_C_ptr, i_ne_offset, ne_mask_copy);
          auto sw_val = mask_gather<sizeof(Scalar)>(Vec(0), inp_slice_C_ptr, i_sw_offset, sw_mask_copy);
          auto se_val = mask_gather<sizeof(Scalar)>(Vec(0), inp_slice_C_ptr, i_se_offset, se_mask_copy);

          auto interpolated = (nw_val * nw) + (ne_val * ne) + (sw_val * sw) + (se_val * se);
          interpolated.store(out_slice[c].data() + offset, len);
        }
        */
    }
    
    #[inline] pub fn bilinear2_backward(&self, 
        g_inp_slice:  &mut TensorAccessor<Scalar,3>,
        g_grid_slice: &mut TensorAccessor<Scalar,3>,
        g_out_slice:  &TensorAccessor<Scalar,3>,
        inp_slice:    &TensorAccessor<Scalar,3>,
        offset:       i64,
        grid_x:       &Vec,
        grid_y:       &Vec,
        len:          i64)  {
        
        todo!();
        /*
            Vec x, y, gx_mult, gy_mult;
        tie(x, gx_mult) = compute_W.apply_get_grad(grid_x);
        tie(y, gy_mult) = compute_H.apply_get_grad(grid_y);

        Vec n, s, w, e, nw, ne, sw, se, nw_mask, ne_mask, sw_mask, se_mask;
        iVec i_y_n, i_x_w;

        tie(
          n, s, w, e, nw, ne, sw, se, nw_mask, ne_mask, sw_mask, se_mask,
          i_y_n, i_x_w) = compute_interp_params(x, y);

        auto i_nw_offset = i_y_n * iVec(inp_sH) + i_x_w * iVec(inp_sW);
        auto i_ne_offset = i_nw_offset + iVec(inp_sW);
        auto i_sw_offset = i_nw_offset + iVec(inp_sH);
        auto i_se_offset = i_sw_offset + iVec(inp_sW);

        auto i_gInp_nw_offset = i_y_n * iVec(inp_W) + i_x_w;
        auto i_gInp_ne_offset = i_gInp_nw_offset + iVec(1);
        auto i_gInp_sw_offset = i_gInp_nw_offset + iVec(inp_W);
        auto i_gInp_se_offset = i_gInp_sw_offset + iVec(1);

        // When reading input values, we used mask_gather. Unfortunately, there is
        // no mask_scatter_add (the backward of mask_gather) in Intel intrinsics.
        // So we store the necessary vectors to temporary arrays and use the helper
        // mask_scatter_add defined above.

        integer_t i_gInp_nw_offset_arr[iVec::size()];
        integer_t i_gInp_ne_offset_arr[iVec::size()];
        integer_t i_gInp_sw_offset_arr[iVec::size()];
        integer_t i_gInp_se_offset_arr[iVec::size()];
        i_gInp_nw_offset.store(i_gInp_nw_offset_arr);
        i_gInp_ne_offset.store(i_gInp_ne_offset_arr);
        i_gInp_sw_offset.store(i_gInp_sw_offset_arr);
        i_gInp_se_offset.store(i_gInp_se_offset_arr);

        integer_t i_nw_mask_arr[iVec::size()];
        integer_t i_ne_mask_arr[iVec::size()];
        integer_t i_sw_mask_arr[iVec::size()];
        integer_t i_se_mask_arr[iVec::size()];
        nw_mask.store(i_nw_mask_arr);
        ne_mask.store(i_ne_mask_arr);
        sw_mask.store(i_sw_mask_arr);
        se_mask.store(i_se_mask_arr);

        Scalar gInp_corner_arr[Vec::size()];

        auto gx = Vec(0), gy = Vec(0);
        #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (i64 c = 0; c < C; ++c) {
          auto inp_slice_C_ptr = inp_slice[c].data();
          auto gInp_slice_C_ptr = gInp_slice[c].data();
          auto gOut = Vec::loadu(gOut_slice[c].data() + offset, len);

          (nw * gOut).store(gInp_corner_arr);
          mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_nw_offset_arr, i_nw_mask_arr, len);
          (ne * gOut).store(gInp_corner_arr);
          mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_ne_offset_arr, i_ne_mask_arr, len);
          (sw * gOut).store(gInp_corner_arr);
          mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_sw_offset_arr, i_sw_mask_arr, len);
          (se * gOut).store(gInp_corner_arr);
          mask_scatter_add(gInp_corner_arr, gInp_slice_C_ptr, i_gInp_se_offset_arr, i_se_mask_arr, len);

          // mask_gather zeros out the mask, so we need to make copies
          Vec nw_mask_copy = nw_mask;
          Vec ne_mask_copy = ne_mask;
          Vec sw_mask_copy = sw_mask;
          Vec se_mask_copy = se_mask;
          auto nw_val = mask_gather<sizeof(Scalar)>(Vec(0), inp_slice_C_ptr, i_nw_offset, nw_mask_copy);
          auto ne_val = mask_gather<sizeof(Scalar)>(Vec(0), inp_slice_C_ptr, i_ne_offset, ne_mask_copy);
          auto sw_val = mask_gather<sizeof(Scalar)>(Vec(0), inp_slice_C_ptr, i_sw_offset, sw_mask_copy);
          auto se_val = mask_gather<sizeof(Scalar)>(Vec(0), inp_slice_C_ptr, i_se_offset, se_mask_copy);

          gx = gx + ((ne_val - nw_val) * s + (se_val - sw_val) * n) * gOut;
          gy = gy + ((sw_val - nw_val) * e + (se_val - ne_val) * w) * gOut;
        }

        gx = gx * gx_mult;
        gy = gy * gy_mult;

        constexpr i64 step = Vec::size();
        auto interleaved_gGrid = interleave2(gx, gy);
        auto gGrid_ptr = gGrid_slice.data() + offset * 2;
        get<0>(interleaved_gGrid).store(gGrid_ptr,
                                             min(len * 2, step));
        get<1>(interleaved_gGrid).store(gGrid_ptr + step,
                                             max(static_cast<i64>(0), len * 2 - step));
        */
    }
    
    pub fn new_nearest2(input: &TensorAccessor<Scalar,4>) -> Self {
    
        todo!();
        /*


            : inp_H(input.size(2))
        , inp_W(input.size(3))
        , inp_sH(input.stride(2))
        , inp_sW(input.stride(3))
        , C(input.size(1))
        , inp_sC(input.stride(1))
        , compute_H(input.size(2))
        , compute_W(input.size(3))
        */
    }
    
    #[inline] pub fn nearest2_forward(&self, 
        out_slice: &mut TensorAccessor<Scalar,3>,
        inp_slice: &TensorAccessor<Scalar,3>,
        offset:    i64,
        grid_x:    &Vec,
        grid_y:    &Vec,
        len:       i64)  {
        
        todo!();
        /*
            auto x = compute_W.apply(grid_x);
        auto y = compute_H.apply(grid_y);

        auto x_nearest = x.round();
        auto y_nearest = y.round();

        auto i_x_nearest = convert_to_int_of_same_size(x_nearest);
        auto i_y_nearest = convert_to_int_of_same_size(y_nearest);

        auto i_mask = must_in_bound ? iVec(-1)
                                    : (i_x_nearest > iVec(-1)) & (i_x_nearest < iVec(inp_W)) &
                                      (i_y_nearest > iVec(-1)) & (i_y_nearest < iVec(inp_H));
        auto mask = cast<Scalar>(i_mask);

        auto i_offset = i_y_nearest * iVec(inp_sH) + i_x_nearest * iVec(inp_sW);

        auto out_ptr = out_slice.data() + offset;
        auto out_sC = out_slice.stride(0);
        auto inp_slice_ptr = inp_slice.data();
        #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (i64 c = 0; c < C; ++c, out_ptr += out_sC, inp_slice_ptr += inp_sC) {
          // mask_gather zeros out the mask, so we need to make a copy
          auto mask_copy = mask;
          auto inp_val = mask_gather<sizeof(Scalar)>(Vec(0), inp_slice_ptr, i_offset, mask_copy);
          inp_val.store(static_cast<void*>(out_ptr), len);
        }
        */
    }
    
    #[inline] pub fn nearest2_backward(&self, 
        g_inp_slice:  &mut TensorAccessor<Scalar,3>,
        g_grid_slice: &mut TensorAccessor<Scalar,3>,
        g_out_slice:  &TensorAccessor<Scalar,3>,
        inp_slice:    &TensorAccessor<Scalar,3>,
        offset:       i64,
        grid_x:       &Vec,
        grid_y:       &Vec,
        len:          i64)  {
        
        todo!();
        /*
            auto x = compute_W.apply(grid_x);
        auto y = compute_H.apply(grid_y);

        auto x_nearest = x.round();
        auto y_nearest = y.round();

        auto i_x_nearest = convert_to_int_of_same_size(x_nearest);
        auto i_y_nearest = convert_to_int_of_same_size(y_nearest);

        auto i_mask = must_in_bound ? iVec(-1)
                                    : (i_x_nearest > iVec(-1)) & (i_x_nearest < iVec(inp_W)) &
                                      (i_y_nearest > iVec(-1)) & (i_y_nearest < iVec(inp_H));

        auto i_gInp_offset = i_y_nearest * iVec(inp_W) + i_x_nearest;  // gInp is contiguous

        integer_t mask_arr[iVec::size()];
        i_mask.store(mask_arr);
        integer_t gInp_offset_arr[iVec::size()];
        i_gInp_offset.store(gInp_offset_arr);

        #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (i64 c = 0; c < C; ++c) {
          mask_scatter_add(gOut_slice[c].data() + offset, gInp_slice[c].data(),
                           gInp_offset_arr, mask_arr, len);
        }

        // grid has zero 0 gradient in Nearest mode
        auto gGrid_ptr = gGrid_slice.data() + offset * 2;
        memset(gGrid_ptr, 0, sizeof(Scalar) * len * 2);
        */
    }
    
    pub fn new_bicubic2(input: &TensorAccessor<Scalar,4>) -> Self {
    
        todo!();
        /*


            : inp_H(input.size(2))
        , inp_W(input.size(3))
        , inp_sH(input.stride(2))
        , inp_sW(input.stride(3))
        , C(input.size(1))
        , inp_sC(input.stride(1))
        , compute_H(input.size(2))
        , compute_W(input.size(3))
        */
    }
    
    /// Calculate the cubic convolution coefficient
    ///
    ///
    #[inline] pub fn bicubic2_get_cubic_coefficients(&self, 
        coeffs: [&Vec<Scalar>; 4],
        tx:     &Vec<Scalar>)
    {
        
        todo!();
        /*
            Vec x;
        x = tx + Vec(1);  // 1 < x = |-1 - tx| < 2
        coeffs[0] = ((A * x - Vec(5) * A) * x + Vec(8) * A) * x - Vec(4) * A;
        x = tx;           // x = |0 - tx| <= 1
        coeffs[1] = ((A + Vec(2)) * x - (A + Vec(3))) * x * x + Vec(1);
        x = Vec(1) - tx;  // x = |1 - tx| <= 1
        coeffs[2] = ((A + Vec(2)) * x - (A + Vec(3))) * x * x + Vec(1);
        x = Vec(2) - tx;  // 1 < x = |2 - tx| < 2
        coeffs[3] = ((A * x - Vec(5) * A) * x + Vec(8) * A) * x - Vec(4) * A;
        */
    }

    /**
      | Calculate the differential of the cubic
      | convolution, i.e. `d coeff / d x`
      |
      */
    #[inline] pub fn bicubic2_get_cubic_coefficients_grad(
        &self, 
        coeffs: [&Vec<Scalar>; 4],
        tx:     &Vec)  {
        
        todo!();
        /*
            Vec x;
        x = Vec(-1) - tx; // 1 < x = |-1 - tx| < 2
        coeffs[0] = (Vec(-3) * A * x - Vec(10) * A ) * x - Vec(8) * A;
        x = Vec(0) - tx;  // x = |0 - tx| <= 1
        coeffs[1] = (Vec(-3) * (A + Vec(2)) * x - Vec(2) * (A + Vec(3))) * x;
        x = Vec(1) - tx;  // x = |1 - tx| <= 1
        coeffs[2] = (Vec(3) * (A + Vec(2)) * x - Vec(2) * (A + Vec(3))) * x;
        x = Vec(2) - tx;  // 1 < x = |2 - tx| < 2
        coeffs[3] = (Vec(3) * A * x - Vec(10) * A) * x + Vec(8) * A;
        */
    }
    
    #[inline] pub fn bicubic2_get_value_bounded(&self, 
        data: *const Scalar,
        x:    &Vec,
        y:    &Vec) -> Vec {
        
        todo!();
        /*
            auto ix = convert_to_int_of_same_size(compute_W.compute_coordinates(x));
        auto iy = convert_to_int_of_same_size(compute_H.compute_coordinates(y));

        auto mask_x = must_in_bound ? iVec(-1) : (ix > iVec(-1)) & (ix < iVec(inp_W));
        auto mask_y = must_in_bound ? iVec(-1) : (iy > iVec(-1)) & (iy < iVec(inp_H));
        auto mask = cast<Scalar>(mask_x & mask_y);

        auto offset = iy * iVec(inp_sH) + ix * iVec(inp_sW);

        auto val = mask_gather<sizeof(Scalar)>(Vec(0), data, offset, mask);
        return val;
        */
    }
    
    #[inline] pub fn bicubic2_add_value_bounded(&self, 
        data:  *mut Scalar,
        len:   i64,
        x:     &Vec,
        y:     &Vec,
        delta: &Vec)  {
        
        todo!();
        /*
            auto ix = convert_to_int_of_same_size(compute_W.compute_coordinates(x));
        auto iy = convert_to_int_of_same_size(compute_H.compute_coordinates(y));

        auto mask_x = must_in_bound ? iVec(-1) : (ix > iVec(-1)) & (ix < iVec(inp_W));
        auto mask_y = must_in_bound ? iVec(-1) : (iy > iVec(-1)) & (iy < iVec(inp_H));
        auto mask = cast<Scalar>(mask_x & mask_y);

        auto i_gInp_offset = iy * iVec(inp_W) + ix;
        integer_t i_gInp_offset_arr[iVec::size()];
        i_gInp_offset.store(i_gInp_offset_arr);

        integer_t mask_arr[iVec::size()];
        mask.store(mask_arr);

        Scalar gInp_corner_arr[Vec::size()];
        delta.store(gInp_corner_arr);

        mask_scatter_add(gInp_corner_arr, data, i_gInp_offset_arr, mask_arr, len);
        */
    }
    
    #[inline] pub fn bicubic2_forward(&self, 
        out_slice: &mut TensorAccessor<Scalar,3>,
        inp_slice: &TensorAccessor<Scalar,3>,
        offset:    i64,
        grid_x:    &Vec,
        grid_y:    &Vec,
        len:       i64)  {
        
        todo!();
        /*
            auto x = compute_W.unnormalize(grid_x);
        auto y = compute_H.unnormalize(grid_y);

        auto ix = x.floor();
        auto iy = y.floor();

        Vec coeff_x[4];
        Vec coeff_y[4];
        get_cubic_coefficients(coeff_x, x - ix);
        get_cubic_coefficients(coeff_y, y - iy);

        #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (i64 c = 0; c < C; ++c) {
          auto inp_slice_C_ptr = inp_slice[c].data();

          // Interpolate the 4 values in the x direction
          Vec interp_x[4];
          for (i64 i = 0; i < 4; ++i) {
            interp_x[i] =
              coeff_x[0] * get_value_bounded(inp_slice_C_ptr, ix - Vec(1), iy + Vec(-1 + i)) +
              coeff_x[1] * get_value_bounded(inp_slice_C_ptr, ix + Vec(0), iy + Vec(-1 + i)) +
              coeff_x[2] * get_value_bounded(inp_slice_C_ptr, ix + Vec(1), iy + Vec(-1 + i)) +
              coeff_x[3] * get_value_bounded(inp_slice_C_ptr, ix + Vec(2), iy + Vec(-1 + i));
          }

          // Interpolate the 4 values in the y direction
          auto interpolated = coeff_y[0] * interp_x[0] + coeff_y[1] * interp_x[1] +
                              coeff_y[2] * interp_x[2] + coeff_y[3] * interp_x[3];
          interpolated.store(out_slice[c].data() + offset, len);
        }
        */
    }
    
    #[inline] pub fn bicubic2_backward(&self, 
        g_inp_slice:  &mut TensorAccessor<Scalar,3>,
        g_grid_slice: &mut TensorAccessor<Scalar,3>,
        g_out_slice:  &TensorAccessor<Scalar,3>,
        inp_slice:    &TensorAccessor<Scalar,3>,
        offset:       i64,
        grid_x:       &Vec,
        grid_y:       &Vec,
        len:          i64)  {
        
        todo!();
        /*
            Vec x = compute_W.unnormalize(grid_x);
        Vec y = compute_H.unnormalize(grid_y);
        Vec gx_mult = Vec(compute_W.scaling_factor);
        Vec gy_mult = Vec(compute_H.scaling_factor);

        auto ix = x.floor();
        auto iy = y.floor();

        Vec coeff_x[4];
        Vec coeff_y[4];
        get_cubic_coefficients(coeff_x, x - ix);
        get_cubic_coefficients(coeff_y, y - iy);

        Vec coeff_x_grad[4];
        Vec coeff_y_grad[4];
        get_cubic_coefficients_grad(coeff_x_grad, x - ix);
        get_cubic_coefficients_grad(coeff_y_grad, y - iy);

        auto gx = Vec(0), gy = Vec(0);
        #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (i64 c = 0; c < C; ++c) {
          auto inp_slice_C_ptr = inp_slice[c].data();
          auto gInp_slice_C_ptr = gInp_slice[c].data();
          auto gOut = Vec::loadu(gOut_slice[c].data() + offset, len);

          for (i64 i = 0; i < 4; ++i) {
            for (i64 j = 0; j < 4; ++j) {
              auto xx = ix + Vec(-1 + i);
              auto yy = iy + Vec(-1 + j);

              add_value_bounded(gInp_slice_C_ptr, len, xx, yy, gOut * coeff_x[i] * coeff_y[j]);

              auto val = get_value_bounded(inp_slice_C_ptr, xx, yy);
              gx = gx - val * gOut * coeff_x_grad[i] * coeff_y[j];
              gy = gy - val * gOut * coeff_y_grad[j] * coeff_x[i];
            }
          }
        }

        gx = gx * gx_mult;
        gy = gy * gy_mult;

        constexpr i64 step = Vec::size();
        auto interleaved_gGrid = interleave2(gx, gy);
        auto gGrid_ptr = gGrid_slice.data() + offset * 2;
        get<0>(interleaved_gGrid).store(gGrid_ptr,
                                             min(len * 2, step));
        get<1>(interleaved_gGrid).store(gGrid_ptr + step,
                                             max(static_cast<i64>(0), len * 2 - step));
        */
    }
}

// ~~~~~~~~~~~~~~~~~~ grid_sample_2d_grid_slice_iterator ~~~~~~~~~~~~~~~~~~~~~~

/**
  | Function to apply a vectorized function on
  | a grid slice tensor (without batch dimension).
  |
  | See NOTE [ Grid Sample CPU Kernels ] for
  | details.
  */
#[inline] pub fn grid_sample_2d_grid_slice_iterator<Scalar, ApplyFn>(
        grid_slice: &TensorAccessor<Scalar,3>,
        apply_fn:   &ApplyFn)  {

    todo!();
        /*
            i64 out_H = grid_slice.size(0);
      i64 out_W = grid_slice.size(1);
      i64 grid_sH = grid_slice.stride(0);
      i64 grid_sW = grid_slice.stride(1);
      i64 grid_sCoor = grid_slice.stride(2);
      auto grid_ptr = grid_slice.data();

      using Vec = Vectorized<Scalar>;
      using iVec = Vectorized<int_same_size_t<Scalar>>;
      constexpr i64 step = Vec::size();

      // Loop over each output pixel in grid.
      // We consider the following three cases (after slicing out the batch
      // dimension).
      // See detailed discussions under each if-case.

      if (geometry_is_contiguous({out_H, out_W, 2}, {grid_sH, grid_sW, grid_sCoor})) {
        // Case 1:
        // Grid is contiguous.
        // Strategy: Sequentially load two vectors at the same time, and get,
        //           e.g.,  {x0, y0, x1, y1}, {x2, y2, x3, y3}. Then we use
        //           vec::deinterleave2 to get x and y vectors.
        auto total_size = out_H * out_W;
        for (i64 spatial_offset = 0; spatial_offset < total_size; spatial_offset += step) {
          auto grid_offset = spatial_offset * 2;
          auto len = min(step, total_size - spatial_offset);
          auto vec1 = Vec::loadu(grid_ptr + grid_offset,
                                 min(step, len * 2));
          auto vec2 = Vec::loadu(grid_ptr + grid_offset + step,
                                 max(static_cast<i64>(0), len * 2 - step));
          auto vec_xy_pair = deinterleave2(vec1, vec2);

          auto x = get<0>(vec_xy_pair);
          auto y = get<1>(vec_xy_pair);

          // make sure that x and y are valid grid sample locations
          if (len < step) {
            x = Vec::set(Vec(0), x, len);
            y = Vec::set(Vec(0), y, len);
          }
          apply_fn(x, y, spatial_offset, len);
        }
      } else if (grid_sW == 1 || out_W == 1) {
        // Case 2:
        // The W dimension is contiguous.
        // This can be common, e.g., grid is from a conv net output of shape
        // [N, 2, H, W].
        // Strategy: Divide into two contiguous slices each of shape [H, W], and
        //           each containing x and y vectors. So we sequentially load a
        //           vector from each of them to get x and y vector

        // Function to apply along a contiguous W dimension (or flattened H x W).
        auto line_fn = [&](const Scalar *grid_ptr_x, const Scalar *grid_ptr_y,
                           i64 out_base_offset, i64 total_size) {
          for (i64 i = 0; i < total_size; i += step) {
            auto len = min(step, total_size - i);
            auto x = Vec::loadu(grid_ptr_x + i, len);
            auto y = Vec::loadu(grid_ptr_y + i, len);
            // make sure that x and y are valid grid sample locations
            if (len < step) {
              x = Vec::set(Vec(0), x, len);
              y = Vec::set(Vec(0), y, len);
            }
            apply_fn(x, y, out_base_offset + i, len);
          }
        };

        if (geometry_is_contiguous({out_H, out_W}, {grid_sH, grid_sW})) {
          // If [H, W] is contiguous, apply line_fn once.
          line_fn(grid_ptr, grid_ptr + grid_sCoor, 0, out_H * out_W);
        } else {
          // If only [W] is contiguous, apply line_fn once for each h slice.
          auto grid_ptr_NH = grid_ptr;
          for (i64 h = 0; h < out_H; h++) {
            line_fn(grid_ptr_NH, grid_ptr_NH + grid_sCoor, h * out_W, out_W);
            grid_ptr_NH += grid_sH;
          }
        }
      } else {
        // Case 3:
        // General case.
        // Strategy: Do a for-loop over H, for each W slice, use
        //           vec::gather to load the x and y vectors.
        i64 spatial_offset = 0;
        const i64 i_offset_delta = grid_sW * step;

        #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
        for (i64 h = 0; h < out_H; h++) {
          auto grid_ptr_x = grid_ptr + h * grid_sH;
          auto grid_ptr_y = grid_ptr_x + grid_sCoor;
          auto i_offsets = iVec::arange(0, grid_sW);
          #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
          # pragma unroll
          #endif
          for (i64 w = 0; w < out_W; w += step) {
            auto len = min(step, out_W - w);
            if (len < step) {
              // prevents illegal memory access, sets the exceeding offsets to zero
              i_offsets = iVec::set(iVec(0), i_offsets, len);
            }
            apply_fn(vec::gather<sizeof(Scalar)>(grid_ptr_x, i_offsets),
                     vec::gather<sizeof(Scalar)>(grid_ptr_y, i_offsets),
                     spatial_offset, len);

            grid_ptr_x += i_offset_delta;
            grid_ptr_y += i_offset_delta;
            spatial_offset += len;
          }
        }
      }
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~ Grid Sample Kernels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
  | Use the structs & functions defined above to
  | calculate grid sample forward and backward.
  |
  | See NOTE [ Grid Sample CPU Kernels ] for
  | details.
  */
pub fn grid_sampler_2d_cpu_kernel_impl(
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: i64,
        padding_mode:       i64,
        align_corners:      bool) -> Tensor {
    
    todo!();
        /*
            auto N = input.size(0);
      auto H = grid.size(1);
      auto W = grid.size(2);
      auto output = empty({N, input.size(1), H, W}, input.options());
      auto spatial_size = H * W;
      auto grain_size = spatial_size == 0 ? (N + 1)
                                          : divup(internal::GRAIN_SIZE, spatial_size * 4 /* 2d * 2 tensors*/);

    #define HANDLE_CASE(interp, padding, align_corners)                            \
      case padding: {                                                              \
        ApplyGridSample<Scalar, 2, interp, padding, align_corners>               \
        grid_sample(inp_acc);                                                      \
        parallel_for(0, N, grain_size, [&](i64 begin, i64 end) {           \
          for (i64 n = begin; n < end; n++) {                                  \
            auto out_slice = out_acc[n];                                           \
            auto inp_slice = inp_acc[n];                                           \
            grid_sample_2d_grid_slice_iterator(                                    \
              grid_acc[n],                                                         \
              [&](const Vectorized<Scalar>& grid_x, const Vectorized<Scalar>& grid_y,  \
                  i64 spatial_offset, i64 len) {                           \
                grid_sample.forward(out_slice, inp_slice, spatial_offset,          \
                                    grid_x, grid_y, len);                          \
              });                                                                  \
            }                                                                      \
          });                                                                      \
        return;                                                                    \
      }

    #define HANDLE_INTERP(interp, align_corners)                                   \
      case interp: {                                                               \
        switch (static_cast<GridSamplerPadding>(padding_mode)) {                   \
          HANDLE_CASE(interp, GridSamplerPadding::Zeros, align_corners);           \
          HANDLE_CASE(interp, GridSamplerPadding::Border, align_corners);          \
          HANDLE_CASE(interp, GridSamplerPadding::Reflection, align_corners);      \
        }                                                                          \
        return;                                                                    \
      }

      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_2d_cpu_kernel_impl", [&] {
        auto out_acc = output.accessor<Scalar, 4>();
        auto inp_acc = input.accessor<Scalar, 4>();
        auto grid_acc = grid.accessor<Scalar, 4>();
        if (align_corners) {
          switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
            HANDLE_INTERP(GridSamplerInterpolation::Bilinear, true);
            HANDLE_INTERP(GridSamplerInterpolation::Nearest, true);
            HANDLE_INTERP(GridSamplerInterpolation::Bicubic, true);
          }
        } else {
          switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
            HANDLE_INTERP(GridSamplerInterpolation::Bilinear, false);
            HANDLE_INTERP(GridSamplerInterpolation::Nearest, false);
            HANDLE_INTERP(GridSamplerInterpolation::Bicubic, false);
          }
        }
      });
    #undef HANDLE_CASE
    #undef HANDLE_INTERP

      return output;
        */
}

pub fn grid_sampler_2d_backward_cpu_kernel_impl(
        grad_output:        &Tensor,
        input:              &Tensor,
        grid:               &Tensor,
        interpolation_mode: i64,
        padding_mode:       i64,
        align_corners:      bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            // grad_output should be contiguous most of time. Ensuring that it is
      // contiguous can greatly simplify this code.
      auto grad_output = grad_output_.contiguous();

      auto grad_input = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto grad_grid = empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto N = input.size(0);
      auto spatial_size = grid.size(1) * grid.size(2);
      auto grain_size = spatial_size == 0 ? (N + 1)
                                          : divup(internal::GRAIN_SIZE, spatial_size * 10 /* 2d * 5 tensors*/);

    #define HANDLE_CASE(interp, padding, align_corners)                              \
      case padding: {                                                                \
        ApplyGridSample<Scalar, 2, interp, padding, align_corners>                 \
        grid_sample(inp_acc);                                                        \
        parallel_for(0, N, grain_size, [&](i64 begin, i64 end) {             \
          for (i64 n = begin; n < end; n++) {                                    \
            auto gInp_slice = gInp_acc[n];                                           \
            auto gGrid_slice = gGrid_acc[n];                                         \
            auto gOut_slice = gOut_acc[n];                                           \
            auto inp_slice = inp_acc[n];                                             \
            grid_sample_2d_grid_slice_iterator(                                      \
              grid_acc[n],                                                           \
              [&](const Vectorized<Scalar>& grid_x, const Vectorized<Scalar>& grid_y,    \
                  i64 spatial_offset, i64 len) {                             \
                grid_sample.backward(gInp_slice, gGrid_slice, gOut_slice, inp_slice, \
                                     spatial_offset, grid_x, grid_y, len);           \
              });                                                                    \
          }                                                                          \
        });                                                                          \
        return;                                                                      \
      }

    #define HANDLE_INTERP(interp, align_corners)                                \
      case interp: {                                                            \
        switch (static_cast<GridSamplerPadding>(padding_mode)) {                \
          HANDLE_CASE(interp, GridSamplerPadding::Zeros, align_corners);        \
          HANDLE_CASE(interp, GridSamplerPadding::Border, align_corners);       \
          HANDLE_CASE(interp, GridSamplerPadding::Reflection, align_corners);   \
        }                                                                       \
        return;                                                                 \
      }

      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_2d_backward_cpu_kernel_impl", [&] {
        auto gInp_acc = grad_input.accessor<Scalar, 4>();
        auto gGrid_acc = grad_grid.accessor<Scalar, 4>();
        auto inp_acc = input.accessor<Scalar, 4>();
        auto grid_acc = grid.accessor<Scalar, 4>();
        auto gOut_acc = grad_output.accessor<Scalar, 4>();
        if (align_corners) {
          switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
            HANDLE_INTERP(GridSamplerInterpolation::Bilinear, true);
            HANDLE_INTERP(GridSamplerInterpolation::Nearest, true);
            HANDLE_INTERP(GridSamplerInterpolation::Bicubic, true);
          }
        } else {
          switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
            HANDLE_INTERP(GridSamplerInterpolation::Bilinear, false);
            HANDLE_INTERP(GridSamplerInterpolation::Nearest, false);
            HANDLE_INTERP(GridSamplerInterpolation::Bicubic, false);
          }
        }
      });
    #undef HANDLE_CASE
    #undef HANDLE_INTERP

      return make_tuple(grad_input, grad_grid);
        */
}

register_dispatch!{grid_sampler_2d_cpu_kernel          , &grid_sampler_2d_cpu_kernel_impl}
register_dispatch!{grid_sampler_2d_backward_cpu_kernel , &grid_sampler_2d_backward_cpu_kernel_impl}
