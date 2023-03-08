crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    OperatorDef,
    CPUContext,
};

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RoIAlignGradientOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    spatial_scale:  f32,
    pooled_height:  i32,
    pooled_width:   i32,
    sampling_ratio: i32,
    aligned:        bool,

    /**
      | Input: X, rois, dY (aka "gradOutput");
      | 
      | Output: dX (aka "gradInput")
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{RoIAlignGradient, 3}

num_outputs!{RoIAlignGradient, 1}

inputs!{RoIAlignGradient, 
    0 => ("X", "See RoIPoolF."),
    1 => ("RoIs", "See RoIPoolF."),
    2 => ("dY", "Gradient of forward output 0 (Y)")
}

outputs!{RoIAlignGradient, 
    0 => ("dX", "Gradient of forward input 0 (X)")
}

impl<T, Context> RoIAlignGradientOp<T, Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            spatial_scale_( this->template GetSingleArgument<float>("spatial_scale", 1.)),
            pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
            pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
            sampling_ratio_( this->template GetSingleArgument<int>("sampling_ratio", -1)),
            aligned_(this->template GetSingleArgument<bool>("aligned", false)) 

        DCHECK_GT(spatial_scale_, 0);
        DCHECK_GT(pooled_height_, 0);
        DCHECK_GT(pooled_width_, 0);
        DCHECK_GE(sampling_ratio_, 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }
}

#[inline] pub fn bilinear_interpolate_gradient<T>(
    height:   i32,
    width:    i32,
    y:        T,
    x:        T,
    w1:       &mut T,
    w2:       &mut T,
    w3:       &mut T,
    w4:       &mut T,
    x_low:    &mut i32,
    x_high:   &mut i32,
    y_low:    &mut i32,
    y_high:   &mut i32,

    /* index for debug only*/
    index:    i32 ) 
{
    todo!();
    /*
        // deal with cases that inverse elements are out of feature map boundary
      if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
      }

      if (y <= 0) {
        y = 0;
      }
      if (x <= 0) {
        x = 0;
      }

      y_low = (int)y;
      x_low = (int)x;

      if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T)y_low;
      } else {
        y_high = y_low + 1;
      }

      if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T)x_low;
      } else {
        x_high = x_low + 1;
      }

      T ly = y - y_low;
      T lx = x - x_low;
      T hy = 1. - ly, hx = 1. - lx;

      // reference in forward
      // T v1 = bottom_data[y_low * width + x_low];
      // T v2 = bottom_data[y_low * width + x_high];
      // T v3 = bottom_data[y_high * width + x_low];
      // T v4 = bottom_data[y_high * width + x_high];
      // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

      w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

      return;
    */
}

#[inline] pub fn add<T>(val: &T, address: *mut T) {
    todo!();
    /*
        *address += val;
    */
}

#[inline] pub fn roi_align_backward_feature<T>(
    nthreads:               i32,
    top_diff:               *const T,
    num_rois:               i32,
    spatial_scale:          &T,
    channels:               i32,
    height:                 i32,
    width:                  i32,
    pooled_height:          i32,
    pooled_width:           i32,
    sampling_ratio:         i32,
    bottom_diff:            *mut T,
    bottom_rois:            *const T,
    rois_cols:              i32,
    continuous_coordinate:  bool) 
{
    todo!();
    /*
        DCHECK(rois_cols == 4 || rois_cols == 5);

      for (int index = 0; index < nthreads; index++) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        const T* offset_bottom_rois = bottom_rois + n * rois_cols;
        int roi_batch_ind = 0;
        if (rois_cols == 5) {
          roi_batch_ind = offset_bottom_rois[0];
          offset_bottom_rois++;
        }

        // Do not using rounding; this implementation detail is critical
        T roi_offset = continuous_coordinate ? T(0.5) : 0;
        T roi_start_w = offset_bottom_rois[0] * spatial_scale - roi_offset;
        T roi_start_h = offset_bottom_rois[1] * spatial_scale - roi_offset;
        T roi_end_w = offset_bottom_rois[2] * spatial_scale - roi_offset;
        T roi_end_h = offset_bottom_rois[3] * spatial_scale - roi_offset;

        T roi_width = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;
        if (continuous_coordinate) {
          CAFFE_ENFORCE(
              roi_width >= 0 && roi_height >= 0,
              "ROIs in ROIAlign do not have non-negative size!");
        } else { // backward compatibility
          // Force malformed ROIs to be 1x1
          roi_width = std::max(roi_width, (T)1.);
          roi_height = std::max(roi_height, (T)1.);
        }
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        T* offset_bottom_diff =
            bottom_diff + (roi_batch_ind * channels + c) * height * width;

        int top_offset = (n * channels + c) * pooled_height * pooled_width;
        const T* offset_top_diff = top_diff + top_offset;
        const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
            ? sampling_ratio
            : ceil(roi_height / pooled_height); // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

        // We do average (integral) pooling inside a bin
        const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
          const T y = roi_start_h + ph * bin_size_h +
              static_cast<T>(iy + .5f) * bin_size_h /
                  static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
          for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            const T x = roi_start_w + pw * bin_size_w +
                static_cast<T>(ix + .5f) * bin_size_w /
                    static_cast<T>(roi_bin_grid_w);

            T w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;

            bilinear_interpolate_gradient(
                height,
                width,
                y,
                x,
                w1,
                w2,
                w3,
                w4,
                x_low,
                x_high,
                y_low,
                y_high,
                index);

            T g1 = top_diff_this_bin * w1 / count;
            T g2 = top_diff_this_bin * w2 / count;
            T g3 = top_diff_this_bin * w3 / count;
            T g4 = top_diff_this_bin * w4 / count;

            if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
              // atomic add is not needed for now since it is single threaded
              add(static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
              add(static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
              add(static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
              add(static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
            } // if
          } // ix
        } // iy
      } // for
    */
}

impl RoIAlignGradientOp<f32, CPUContext> {

    #[inline] pub fn run_f32_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // Input data to pool
      auto& R = Input(1); // RoIs
      auto& dY = Input(2); // Gradient of net w.r.t. output of "forward" op
                           // (aka "gradOutput")

      CAFFE_ENFORCE_EQ(R.dim(), 2);
      // if R has 5 columns, the first column is the index, otherwise 0
      CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);

      auto* dX = Output(
          0,
          X.sizes(),
          at::dtype<float>()); // Gradient of net w.r.t. input to "forward" op (aka
                               // "gradInput")

      // Must zero-out dX before accumulating gradients
      // (TODO): Kaiming - is this safe?
      math::Set<float, CPUContext>(
          dX->numel(), 0.f, dX->template mutable_data<float>(), &context_);

      if (dY.numel() > 0) { // Handle possibly empty gradient if there were no rois
        ROIAlignBackwardFeature<float>(
            dY.numel(),
            dY.data<float>(),
            R.dim32(0),
            spatial_scale_,
            X.dim32(1),
            X.dim32(2),
            X.dim32(3),
            pooled_height_,
            pooled_width_,
            sampling_ratio_,
            dX->template mutable_data<float>(),
            R.data<float>(),
            R.dim32(1),
            aligned_);
      }
      return true;
        */
    }
}

register_cpu_operator!{RoIAlignGradient, RoIAlignGradientOp<float, CPUContext>}

pub struct GetRoIAlignGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRoIAlignGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RoIAlignGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{RoIAlign, GetRoIAlignGradient}

pub type RoIAlignGradientCPUOp<T> = RoIAlignGradientOp<T, CPUContext>;
