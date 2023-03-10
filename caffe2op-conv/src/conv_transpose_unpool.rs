crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ConvTransposeUnpoolBase<Context> {

    storage: OperatorStorage,
    context: Context,


    legacy_pad:      LegacyPadding,
    pad:             i32,
    kernel:          Vec<i32>,
    stride:          Vec<i32>,
    pads:            Vec<i32>,
    adj:             Vec<i32>,
    group:           i32,
    order:           StorageOrder,
    shared_buffer:   bool,
    ws:              *mut Workspace,
}

impl<Context> ConvTransposeUnpoolBase<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            legacy_pad_(
                static_cast<LegacyPadding>(this->template GetSingleArgument<int>(
                    "legacy_pad",
                    LegacyPadding::NOTSET))),
            kernel_(this->template GetRepeatedArgument<int>("kernels")),
            stride_(this->template GetRepeatedArgument<int>("strides")),
            pads_(this->template GetRepeatedArgument<int>("pads")),
            adj_(this->template GetRepeatedArgument<int>("adjs")),
            group_(this->template GetSingleArgument<int>("group", 1)),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<string>("order", "NCHW"))),
            shared_buffer_(
                this->template GetSingleArgument<int>("shared_buffer", 0)),
            ws_(ws) 


        // For the padding, they should either be the legacy padding strategy
        // (VALID or SAME), or an explicit, non-negative value.
        if (legacy_pad_ == LegacyPadding::VALID ||
            legacy_pad_ == LegacyPadding::SAME) {
          CAFFE_ENFORCE(
              !OperatorStorage::HasArgument("pads"),
              "If you use legacy padding VALID or SAME, you should not specify "
              "any specific padding values.");
        }
        // Get old arguments values.
        if (OperatorStorage::HasArgument("kernel")) {
          kernel_.resize(2, this->template GetSingleArgument<int>("kernel", 0));
        } else if (
            OperatorStorage::HasArgument("kernel_h") &&
            OperatorStorage::HasArgument("kernel_w")) {
          kernel_.push_back(this->template GetSingleArgument<int>("kernel_h", 0));
          kernel_.push_back(this->template GetSingleArgument<int>("kernel_w", 0));
        }

        if (OperatorStorage::HasArgument("stride")) {
          stride_.resize(2, this->template GetSingleArgument<int>("stride", 0));
        } else if (
            OperatorStorage::HasArgument("stride_h") &&
            OperatorStorage::HasArgument("stride_w")) {
          stride_.push_back(this->template GetSingleArgument<int>("stride_h", 0));
          stride_.push_back(this->template GetSingleArgument<int>("stride_w", 0));
        }

        if (OperatorStorage::HasArgument("adj")) {
          adj_.resize(2, this->template GetSingleArgument<int>("adj", 0));
        } else if (
            OperatorStorage::HasArgument("adj_h") &&
            OperatorStorage::HasArgument("adj_w")) {
          adj_.push_back(this->template GetSingleArgument<int>("adj_h", 0));
          adj_.push_back(this->template GetSingleArgument<int>("adj_w", 0));
        }

        if (OperatorStorage::HasArgument("pad")) {
          CAFFE_ENFORCE(
              legacy_pad_ != LegacyPadding::VALID &&
                  legacy_pad_ != LegacyPadding::SAME,
              "If you use legacy padding VALID or SAME, you should not specify "
              "any specific padding values.");
          pads_.resize(4, this->template GetSingleArgument<int>("pad", 0));
        } else if (
            OperatorStorage::HasArgument("pad_t") &&
            OperatorStorage::HasArgument("pad_l") &&
            OperatorStorage::HasArgument("pad_b") &&
            OperatorStorage::HasArgument("pad_r")) {
          CAFFE_ENFORCE(
              legacy_pad_ != LegacyPadding::VALID &&
                  legacy_pad_ != LegacyPadding::SAME,
              "If you use legacy padding VALID or SAME, you should not specify "
              "any specific padding values.");
          pads_.push_back(this->template GetSingleArgument<int>("pad_t", 0));
          pads_.push_back(this->template GetSingleArgument<int>("pad_l", 0));
          pads_.push_back(this->template GetSingleArgument<int>("pad_b", 0));
          pads_.push_back(this->template GetSingleArgument<int>("pad_r", 0));
        }

        // Fill default values.
        if (kernel_.size() == 0) {
          kernel_.assign({0, 0});
        }

        if (stride_.size() == 0) {
          stride_.resize(kernel_.size(), 1);
        }

        if (pads_.size() == 0) {
          pads_.resize(kernel_.size() * 2, 0);
        }

        if (adj_.size() == 0) {
          adj_.resize(kernel_.size(), 0);
        }

        CAFFE_ENFORCE_EQ(stride_.size(), kernel_.size());
        CAFFE_ENFORCE_EQ(adj_.size(), kernel_.size());

        if (legacy_pad_ != LegacyPadding::VALID &&
            legacy_pad_ != LegacyPadding::SAME) {
          CAFFE_ENFORCE_EQ(pads_.size(), 2 * kernel_.size());
        }

        for (int dim = 0; dim < kernel_.size(); ++dim) {
          CAFFE_ENFORCE_GT(kernel_[dim], 0);
          CAFFE_ENFORCE_GT(stride_[dim], 0);
          CAFFE_ENFORCE_GE(adj_[dim], 0);
          CAFFE_ENFORCE_LE(adj_[dim], stride_[dim]);
        }

        // Create shared buffer mutex in the constructor
        // to avoid race-condition in DAGNet.
        if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
          createSharedBuffer<Context>(ws_);
        }
        */
    }

    /**
      | Gets the output size. The output channel
      | is manually specified.
      |
      */
    #[inline] pub fn get_output_size(&mut self, 
        input:          &Tensor, 
        output_channel: i32) -> Vec<i64> {
        
        todo!();
        /*
            CAFFE_ENFORCE(4 == input.dim());
        CAFFE_ENFORCE_GT(input.size_from_dim(1), 0);
        int N = input.dim32(0);
        bool channel_first = false; // initialized to suppress compiler warning.
        int H = 0, W = 0; // initialized to suppress compiler warning.
        int M = 0;
        switch (order_) {
          case StorageOrder::NHWC:
            channel_first = false;
            H = input.dim32(1);
            W = input.dim32(2);
            M = input.dim32(3);
            break;
          case StorageOrder::NCHW:
            channel_first = true;
            M = input.dim32(1);
            H = input.dim32(2);
            W = input.dim32(3);
            break;
          default:
            LOG(FATAL) << "Unknown Storage order: " << order_;
        }
        int output_height = 0, output_width = 0;
        ComputeSizeAndPad(
            H,
            stride_[0],
            kernel_[0],
            adj_[0],
            &pads_[0],
            &pads_[2],
            &output_height);
        ComputeSizeAndPad(
            W,
            stride_[1],
            kernel_[1],
            adj_[1],
            &pads_[1],
            &pads_[3],
            &output_width);
        std::vector<int64_t> sizes;
        if (channel_first) {
          sizes = {N, output_channel, output_height, output_width};
        } else {
          sizes = {N, output_height, output_width, output_channel};
        }
        VLOG(2) << "In: N " << N << " M " << M << " H " << H << " W " << W;
        VLOG(2) << "Out: output_channel " << output_channel << " H "
                << output_height << " W " << output_width;
        return sizes;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            switch (order_) {
          case StorageOrder::NHWC:
            return RunOnDeviceWithOrderNHWC();
          case StorageOrder::NCHW:
            return RunOnDeviceWithOrderNCHW();
          default:
            LOG(FATAL) << "Unknown storage order: " << order_;
        }
        // To suppress old compiler warnings
        return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_THROW("Not implemented");
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_THROW("Not implemented");
        */
    }

    /// ------------------- Accessors for 2D conv params.
    
    #[inline] pub fn pad_t(&self) -> i32 {
        
        todo!();
        /*
            return pads_[0];
        */
    }
    
    #[inline] pub fn pad_l(&self) -> i32 {
        
        todo!();
        /*
            return pads_[1];
        */
    }
    
    #[inline] pub fn pad_b(&self) -> i32 {
        
        todo!();
        /*
            return pads_[2];
        */
    }
    
    #[inline] pub fn pad_r(&self) -> i32 {
        
        todo!();
        /*
            return pads_[3];
        */
    }
    
    #[inline] pub fn kernel_h(&self) -> i32 {
        
        todo!();
        /*
            return kernel_[0];
        */
    }
    
    #[inline] pub fn kernel_w(&self) -> i32 {
        
        todo!();
        /*
            return kernel_[1];
        */
    }
    
    #[inline] pub fn stride_h(&self) -> i32 {
        
        todo!();
        /*
            return stride_[0];
        */
    }
    
    #[inline] pub fn stride_w(&self) -> i32 {
        
        todo!();
        /*
            return stride_[1];
        */
    }
    
    #[inline] pub fn adj_h(&self) -> i32 {
        
        todo!();
        /*
            return adj_[0];
        */
    }
    
    #[inline] pub fn adj_w(&self) -> i32 {
        
        todo!();
        /*
            return adj_[1];
        */
    }
    
    #[inline] pub fn compute_size_and_pad(
        &mut self, 
        in_size:   i32,
        stride:    i32,
        kernel:    i32,
        adj:       i32,
        pad_head:  *mut i32,
        pad_tail:  *mut i32,
        out_size:  *mut i32)  
    {
        
        todo!();
        /*
            switch (legacy_pad_) {
          case LegacyPadding::NOTSET:
            CAFFE_ENFORCE(*pad_head >= 0);
            CAFFE_ENFORCE(*pad_tail >= 0);
            *out_size =
                (in_size - 1) * stride + kernel + adj - *pad_head - *pad_tail;
            break;
          // We handle cases of LegacyPadding::VALID and LegacyPadding::SAME
          // the same way
          case LegacyPadding::VALID:
          case LegacyPadding::SAME:
            *pad_head = 0;
            *pad_tail = 0;
            *out_size = (in_size - 1) * stride + kernel + adj;
            break;
          case LegacyPadding::CAFFE_LEGACY_POOLING:
            LOG(FATAL) << "CAFFE_LEGACY_POOLING is no longer supported.";
            break;
        }
        */
    }
}

