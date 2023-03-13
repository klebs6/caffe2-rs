crate::ix!();


pub struct IDEEPConvTransposeUnpoolBase {
    //USE_IDEEP_DEF_ALIASES();
    //USE_IDEEP_OPERATOR_FUNCTIONS();
    base: IDEEPOperator,

    legacy_pad:     LegacyPadding,
    pad:            i32,
    kernel:         Vec<i32>,
    stride:         Vec<i32>,
    pads:           Vec<i32>,
    adj:            Vec<i32>,
    shared_buffer:  bool,
}

impl IDEEPConvTransposeUnpoolBase {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            legacy_pad_(
                static_cast<LegacyPadding>(OperatorStorage::GetSingleArgument<int>(
                    "legacy_pad",
                    LegacyPadding::NOTSET))),
            kernel_(OperatorStorage::GetRepeatedArgument<int>("kernels")),
            stride_(OperatorStorage::GetRepeatedArgument<int>("strides")),
            pads_(OperatorStorage::GetRepeatedArgument<int>("pads")),
            adj_(OperatorStorage::GetRepeatedArgument<int>("adjs")),
            shared_buffer_( OperatorStorage::GetSingleArgument<int>("shared_buffer", 0)) 

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
          kernel_.resize(2, OperatorStorage::GetSingleArgument<int>("kernel", 0));
        } else if (
            OperatorStorage::HasArgument("kernel_h") &&
            OperatorStorage::HasArgument("kernel_w")) {
          kernel_.push_back(OperatorStorage::GetSingleArgument<int>("kernel_h", 0));
          kernel_.push_back(OperatorStorage::GetSingleArgument<int>("kernel_w", 0));
        }

        if (OperatorStorage::HasArgument("stride")) {
          stride_.resize(2, OperatorStorage::GetSingleArgument<int>("stride", 0));
        } else if (
            OperatorStorage::HasArgument("stride_h") &&
            OperatorStorage::HasArgument("stride_w")) {
          stride_.push_back(OperatorStorage::GetSingleArgument<int>("stride_h", 0));
          stride_.push_back(OperatorStorage::GetSingleArgument<int>("stride_w", 0));
        }

        if (OperatorStorage::HasArgument("adj")) {
          adj_.resize(2, OperatorStorage::GetSingleArgument<int>("adj", 0));
        } else if (
            OperatorStorage::HasArgument("adj_h") &&
            OperatorStorage::HasArgument("adj_w")) {
          adj_.push_back(OperatorStorage::GetSingleArgument<int>("adj_h", 0));
          adj_.push_back(OperatorStorage::GetSingleArgument<int>("adj_w", 0));
        }

        if (OperatorStorage::HasArgument("pad")) {
          CAFFE_ENFORCE(
              legacy_pad_ != LegacyPadding::VALID &&
                  legacy_pad_ != LegacyPadding::SAME,
              "If you use legacy padding VALID or SAME, you should not specify "
              "any specific padding values.");
          pads_.resize(4, OperatorStorage::GetSingleArgument<int>("pad", 0));
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
          pads_.push_back(OperatorStorage::GetSingleArgument<int>("pad_t", 0));
          pads_.push_back(OperatorStorage::GetSingleArgument<int>("pad_l", 0));
          pads_.push_back(OperatorStorage::GetSingleArgument<int>("pad_b", 0));
          pads_.push_back(OperatorStorage::GetSingleArgument<int>("pad_r", 0));
        }

        // Fill default values.
        if (kernel_.empty()) {
          kernel_.assign({0, 0});
        }

        if (stride_.empty()) {
          stride_.assign(kernel_.size(), 1);
        }

        if (pads_.empty()) {
          pads_.assign(kernel_.size() * 2, 0);
        }

        if (adj_.empty()) {
          adj_.assign(kernel_.size(), 0);
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
        */
    }
    
    #[inline] pub fn input(&mut self, index: i32) -> &IDEEPTensor {
        
        todo!();
        /*
            return OperatorStorage::template Input<ideep::tensor>(index);
        */
    }
    
    #[inline] pub fn output(&mut self, index: i32) -> *mut IDEEPTensor {
        
        todo!();
        /*
            return OperatorStorage::template Output<ideep::tensor>(index);
        */
    }
    
    #[inline] pub fn pad_tl(&self) -> IDEEPTensorDims {
        
        todo!();
        /*
            return {pad_t(), pad_l()};
        */
    }
    
    #[inline] pub fn pad_br(&self) -> IDEEPTensorDims {
        
        todo!();
        /*
            return {pad_b(), pad_r()};
        */
    }
    
    #[inline] pub fn calc_output_dims(
        &mut self, 
        input: &IDEEPTensor,
        output_channel: i32) -> IDEEPTensorDims 
    {
        todo!();
        /*
            CAFFE_ENFORCE_GT(input.get_size(), 0);

        int N = input.get_dim(0);
        ideep::tensor::dims output_dims;
        auto input_dims = input.get_dims();
        itensor::dims dims;
        dims.assign(input_dims.begin() + 2, input_dims.end());
        for (int dim = 0; dim < dims.size(); ++dim) {
          int dim_size = 0;
          ComputeSizeAndPad(
              dims[dim],
              stride_[dim],
              kernel_[dim],
              adj_[dim],
              &pads_[dim],
              &pads_[dim + 2],
              &dim_size);
          output_dims.push_back(dim_size);
        }

        output_dims.insert(output_dims.begin(), {N, output_channel});
        return output_dims;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            try {
          return RunOnDeviceWithOrderNCHW();
        } catch (ideep::error& e) {
          LOG(ERROR) << "IDEEP error:" << e.message;
          throw;
        }
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_THROW("Not implemented");
        */
    }
    
    /// Accessors for 2D conv params. --------------------------
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
            CAFFE_ENFORCE_GE(*pad_head, 0);
            CAFFE_ENFORCE_GE(*pad_tail, 0);
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

#[macro_export] macro_rules! use_ideep_conv_transpose_unpool_base_functions {
    () => {
        todo!();
        /*
        USE_OPERATOR_BASE_FUNCTIONS;                                    \
            /* using override */ using IDEEPConvTransposeUnpoolBase::Input; \
            /* using override */ using IDEEPConvTransposeUnpoolBase::Output;
            */
    }
}
