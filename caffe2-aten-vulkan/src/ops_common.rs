crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Common.h]

pub mod layout {

    /// 4D Activation Maps
    pub mod activation4d {
        pub const BATCH:    usize = 0;
        pub const CHANNELS: usize = 1;
        pub const HEIGHT:   usize = 2;
        pub const WIDTH:    usize = 3;
    }

    /// Convolution Filters
    pub mod filter {
        pub const OUTPUT: usize = 0;
        pub const INPUT:  usize = 1;
        pub const HEIGHT: usize = 2;
        pub const WIDTH:  usize = 3;
    }

    /// Parameters (Pooling Kernels, Dilation,
    /// Padding, Stride, etc.)
    ///
    pub mod parameter {
        pub const HEIGHT: usize = 0;
        pub const WIDTH:  usize = 1;
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Common.cpp]

pub fn batch_size(tensor: &Tensor) -> u32 {
    
    todo!();
        /*
            const IntArrayRef sizes = tensor.sizes();
      const u32 dims = sizes.size();
      if (dims < 4) {
        return 1;
      }
      return sizes[dims - 4];
        */
}

pub fn channels_size(tensor: &Tensor) -> u32 {
    
    todo!();
        /*
            const IntArrayRef sizes = tensor.sizes();
      const u32 dims = sizes.size();
      if (dims < 3) {
        return 1;
      }
      return sizes[dims - 3];
        */
}

pub fn height_size(tensor: &Tensor) -> u32 {
    
    todo!();
        /*
            const IntArrayRef sizes = tensor.sizes();
      const u32 dims = sizes.size();
      if (dims < 2) {
        return 1;
      }
      return sizes[dims - 2];
        */
}

pub fn width_size(tensor: &Tensor) -> u32 {
    
    todo!();
        /*
            const IntArrayRef sizes = tensor.sizes();
      const u32 dims = sizes.size();
      if (dims < 1) {
        return 1;
      }
      return sizes[dims - 1];
        */
}
