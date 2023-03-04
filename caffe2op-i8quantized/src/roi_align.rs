crate::ix!();

pub struct PreCalc {
    pos1:  i32,
    pos2:  i32,
    pos3:  i32,
    pos4:  i32,
    w1:    u8,
    w2:    u8,
    w3:    u8,
    w4:    u8,
}

#[inline] pub fn pre_calc_for_bilinear_interpolate(
    height:           i32,
    width:            i32,
    pooled_height:    i32,
    pooled_width:     i32,
    iy_upper:         i32,
    ix_upper:         i32,
    roi_start_h:      f32,
    roi_start_w:      f32,
    bin_size_h:       f32,
    bin_size_w:       f32,
    roi_bin_grid_h:   i32,
    roi_bin_grid_w:   i32,
    pre_calc:         &mut Vec<PreCalc>)
{
    
    todo!();
    /*
        int pre_calc_index = 0;
      // boltnn use a smaller multiplier here. Sometimes w will shrink to 0.
      const float w_multiplier = 255.0;
      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          for (int iy = 0; iy < iy_upper; iy++) {
            const float yy = roi_start_h + ph * bin_size_h +
                static_cast<float>(iy + .5f) * bin_size_h /
                    static_cast<float>(roi_bin_grid_h); // e.g., 0.5, 1.5
            for (int ix = 0; ix < ix_upper; ix++) {
              const float xx = roi_start_w + pw * bin_size_w +
                  static_cast<float>(ix + .5f) * bin_size_w /
                      static_cast<float>(roi_bin_grid_w);

              float x = xx;
              float y = yy;
              // deal with: inverse elements are out of feature map boundary
              if (y < -1.0 || y > height || x < -1.0 || x > width) {
                // empty
                PreCalc pc;
                pc.pos1 = 0;
                pc.pos2 = 0;
                pc.pos3 = 0;
                pc.pos4 = 0;
                pc.w1 = 0;
                pc.w2 = 0;
                pc.w3 = 0;
                pc.w4 = 0;
                pre_calc[pre_calc_index] = pc;
                pre_calc_index += 1;
                continue;
              }

              if (y <= 0) {
                y = 0;
              }
              if (x <= 0) {
                x = 0;
              }

              int y_low = (int)y;
              int x_low = (int)x;
              int y_high;
              int x_high;

              if (y_low >= height - 1) {
                y_high = y_low = height - 1;
                y = (float)y_low;
              } else {
                y_high = y_low + 1;
              }

              if (x_low >= width - 1) {
                x_high = x_low = width - 1;
                x = (float)x_low;
              } else {
                x_high = x_low + 1;
              }

              float ly = y - y_low;
              float lx = x - x_low;
              float hy = 1. - ly, hx = 1. - lx;
              // w are not necessary 1
              uint8_t w1 = static_cast<uint8_t>(Round(hy * hx * w_multiplier));
              uint8_t w2 = static_cast<uint8_t>(Round(hy * lx * w_multiplier));
              uint8_t w3 = static_cast<uint8_t>(Round(ly * hx * w_multiplier));
              uint8_t w4 = static_cast<uint8_t>(Round(ly * lx * w_multiplier));

              // save weights and indeces
              PreCalc pc;
              pc.pos1 = y_low * width + x_low;
              pc.pos2 = y_low * width + x_high;
              pc.pos3 = y_high * width + x_low;
              pc.pos4 = y_high * width + x_high;

              pc.w1 = w1;
              pc.w2 = w2;
              pc.w3 = w3;
              pc.w4 = w4;
              pre_calc[pre_calc_index] = pc;

              pre_calc_index += 1;
            }
          }
        }
      }
    */
}

#[inline] pub fn rOIAlign_forward(
    nthreads:                i32,
    bottom_data:             *const u8,
    spatial_scale:           &f32,
    channels:                i32,
    height:                  i32,
    width:                   i32,
    pooled_height:           i32,
    pooled_width:            i32,
    sampling_ratio:          i32,
    bottom_rois:             *const f32,
    roi_cols:                i32,
    top_data:                *mut u8,
    x_scale:                 f32,
    y_scale:                 f32,
    x_offset:                i32,
    y_offset:                i32,
    order:                   StorageOrder,
    continuous_coordinate:   bool)  
{
    todo!();
    /*
        DCHECK(roi_cols == 4 || roi_cols == 5);

      int n_rois = nthreads / channels / pooled_width / pooled_height;

      for (int n = 0; n < n_rois; n++) {
        int index_n = n * channels * pooled_width * pooled_height;

        // roi could have 4 or 5 columns
        const float* offset_bottom_rois = bottom_rois + n * roi_cols;
        int roi_batch_ind = 0;
        if (roi_cols == 5) {
          roi_batch_ind = offset_bottom_rois[0];
          offset_bottom_rois++;
        }

        // Do not using rounding; this implementation detail is critical
        float roi_offset = continuous_coordinate ? 0.5 : 0;
        float roi_start_w = offset_bottom_rois[0] * spatial_scale - roi_offset;
        float roi_start_h = offset_bottom_rois[1] * spatial_scale - roi_offset;
        float roi_end_w = offset_bottom_rois[2] * spatial_scale - roi_offset;
        float roi_end_h = offset_bottom_rois[3] * spatial_scale - roi_offset;

        float roi_width = roi_end_w - roi_start_w;
        float roi_height = roi_end_h - roi_start_h;
        if (continuous_coordinate) {
          CAFFE_ENFORCE(
              roi_width >= 0 && roi_height >= 0,
              "ROIs in ROIAlign do not have non-negative size!");
        } else { // backward compatibility
          // Force malformed ROIs to be 1x1
          roi_width = std::max(roi_width, (float)1.);
          roi_height = std::max(roi_height, (float)1.);
        }
        float bin_size_h =
            static_cast<float>(roi_height) / static_cast<float>(pooled_height);
        float bin_size_w =
            static_cast<float>(roi_width) / static_cast<float>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
            ? sampling_ratio
            : ceil(roi_height / pooled_height); // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

        // We do average (integral) pooling inside a bin
        const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

        // calculate multiplier
        double real_multiplier = x_scale / (y_scale * 255.0 * count);
        int32_t Y_multiplier;
        int Y_shift;
        QuantizeMultiplierSmallerThanOne(real_multiplier, &Y_multiplier, &Y_shift);

        // we want to precalculate indeces and weights shared by all chanels,
        // this is the key point of optimiation
        std::vector<PreCalc> pre_calc(
            roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        pre_calc_for_bilinear_interpolate(
            height,
            width,
            pooled_height,
            pooled_width,
            roi_bin_grid_h,
            roi_bin_grid_w,
            roi_start_h,
            roi_start_w,
            bin_size_h,
            bin_size_w,
            roi_bin_grid_h,
            roi_bin_grid_w,
            pre_calc);

        const uint8_t* offset_bottom_data =
            bottom_data + roi_batch_ind * channels * height * width;
        int pre_calc_index = 0;
        for (int ph = 0; ph < pooled_height; ph++) {
          for (int pw = 0; pw < pooled_width; pw++) {
            vector<int32_t> acc_buffer(channels, 0);

            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
              for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                PreCalc pc = pre_calc[pre_calc_index];

                const uint8_t* data_1 = offset_bottom_data + channels * pc.pos1;
                const uint8_t* data_2 = offset_bottom_data + channels * pc.pos2;
                const uint8_t* data_3 = offset_bottom_data + channels * pc.pos3;
                const uint8_t* data_4 = offset_bottom_data + channels * pc.pos4;
                for (int c = 0; c < channels; ++c) {
                  acc_buffer[c] += (uint32_t)(pc.w1) * (uint32_t)(data_1[c]);
                  acc_buffer[c] += (uint32_t)(pc.w2) * (uint32_t)(data_2[c]);
                  acc_buffer[c] += (uint32_t)(pc.w3) * (uint32_t)(data_3[c]);
                  acc_buffer[c] += (uint32_t)(pc.w4) * (uint32_t)(data_4[c]);

                  // w_1..4 are all multiplied by 255.0
                  acc_buffer[c] -= x_offset * 255.0;
                }

                pre_calc_index += 1;
              }
            }
            int index_nhw = index_n + (ph * pooled_width + pw) * channels;
            uint8_t* out_ptr = top_data + index_nhw;
            for (int c = 0; c < channels; ++c) {
              int32_t a_mul = MultiplyByQuantizedMultiplierSmallerThanOne(
                                  acc_buffer[c], Y_multiplier, Y_shift) +
                  y_offset;
              int32_t clamped_a =
                  std::min<int32_t>(255, std::max<int32_t>(0, a_mul));
              out_ptr[c] = static_cast<uint8_t>(clamped_a);
            }
          } // for pw
        } // for ph
      } // for n
    */
}

/**
  | Region of Interest (RoI) align operation
  | as used in Mask R-CNN.
  |
  */
pub struct Int8RoIAlignOp {
    
    storage: OperatorStorage,
    context: CPUContext,
    order:          StorageOrder,
    spatial_scale:  f32,
    pooled_height:  i32,
    pooled_width:   i32,
    sampling_ratio: i32,
    aligned:        bool,
}

impl Int8RoIAlignOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<string>("order", "NHWC"))),
            spatial_scale_(
                this->template GetSingleArgument<float>("spatial_scale", 1.)),
            pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
            pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
            sampling_ratio_(
                this->template GetSingleArgument<int>("sampling_ratio", -1)),
            aligned_(this->template GetSingleArgument<bool>("aligned", false)) 

        DCHECK_GT(spatial_scale_, 0);
        DCHECK_GT(pooled_height_, 0);
        DCHECK_GT(pooled_width_, 0);
        DCHECK_GE(sampling_ratio_, 0);
        // only supports NHWC
        CAFFE_ENFORCE(order_ == StorageOrder::NHWC);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Inputs()[0]->template Get<Int8TensorCPU>(); // Input, NHWC
        auto& R = Input(1); // RoIs
        auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>(); // RoI pooled
        // calculate multiplier
        int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
        auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        Y->scale = Y_scale;
        Y->zero_point = Y_offset;

        if (R.numel() == 0) {
          // Handle empty rois
          Y->t.Resize(0, pooled_height_, pooled_width_, X.t.dim32(3));
          // The following mutable_data calls are needed to allocate the tensors
          Y->t.mutable_data<uint8_t>();
          return true;
        }

        CAFFE_ENFORCE_EQ(R.dim(), 2);
        // if R has 5 columns, the first column is the index, otherwise 0
        CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);

        assert(sampling_ratio_ >= 0);

        // only supports NHWC now
        ReinitializeTensor(
            &Y->t,
            {R.dim32(0), pooled_height_, pooled_width_, X.t.dim32(3)},
            at::dtype<uint8_t>().device(CPU));
        int output_size = Y->t.numel();

        ROIAlignForward(
            output_size,
            X.t.data<uint8_t>(),
            spatial_scale_,
            X.t.dim32(3),
            X.t.dim32(1),
            X.t.dim32(2),
            pooled_height_,
            pooled_width_,
            sampling_ratio_,
            R.data<float>(),
            R.dim32(1),
            Y->t.mutable_data<uint8_t>(),
            X.scale,
            Y_scale,
            X.zero_point,
            Y_offset,
            order_,
            aligned_);

        return true;
        */
    }
}

register_cpu_operator!{Int8RoIAlign, int8::Int8RoIAlignOp}

num_inputs!{Int8RoIAlign, 2}

num_outputs!{Int8RoIAlign, 1}

inputs!{Int8RoIAlign, 
    0 => ("X",                "4D Int8 Tensor feature map input of shape (N, C, H, W)."),
    1 => ("RoIs",             "2D input of shape (R, 4 or 5) specifying 
        R RoIs representing: batch index in [0, N - 1], x1, y1, x2, y2. 
        The RoI coordinates are in the coordinate system of the input image. 
        For inputs corresponding to a single image, batch index can be excluded to have 
        just 4 columns.")
}

outputs!{Int8RoIAlign, 
    0 => ("Y",                "4D Int8 Tensor output of shape (R, C, pooled_h, pooled_w). 
        The r-th batch element is a pooled feature map cooresponding to the r-th RoI.")
}

args!{Int8RoIAlign, 
    0 => ("Y_scale",          "Output tensor quantization scale"),
    1 => ("Y_zero_point",     "Output tensor quantization offset"),
    2 => ("spatial_scale",    "(float) default 1.0; Spatial scale of the input feature map 
        X relative to the input image. E.g., 0.0625 if X has a stride of 16 w.r.t. the input image."),
    3 => ("pooled_h",         "(int) default 1; Pooled output Y's height."),
    4 => ("pooled_w",         "(int) default 1; Pooled output Y's width."),
    5 => ("sampling_ratio",   "(int) default -1; number of sampling points in 
        the interpolation grid used to compute the output value of each pooled output bin. 
        If > 0, then exactly sampling_ratio x sampling_ratio grid points are used. 
        If <= 0, then an adaptive number of grid points are used 
        (computed as ceil(roi_width / pooled_w), and likewise for height).")
}

