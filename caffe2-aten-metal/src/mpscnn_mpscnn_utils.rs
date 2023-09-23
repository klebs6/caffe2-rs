crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/metal/mpscnn/MPSCNNUtils.h]

pub struct LaunchParams {
    threads_per_threadgroup: MTLSize,
    threadgroups_per_grid:   MTLSize,

    /**
      | iOS 11.0
      |
      */
    threads_per_grid:        MTLSize,
}

//#[API_AVAILABLE(ios(10.0), macos(10.13))]
#[inline] pub fn kernel_for(
        image:            *mut MPSImage,
        array_kernel:     &String,
        non_array_kernel: &String) -> String {
    
    todo!();
        /*
            if (image.featureChannels > 4 || image.numberOfImages > 1) {
        return arrayKernel;
      }
      return nonArrayKernel;
        */
}

#[inline] pub fn compute_mps_align_offset(
        kernel: i32,
        pad:    i32) -> i32 {
    
    todo!();
        /*
            // To set the offset, we can just match the top-left pixel (in the input
      // image, with negative values for padding) that we look at. For 3x3s1p1, we
      // look at the (-1, -1) pixel in the original impl. For 3x3s1p0, we look at
      // (0, 0) pixel. For 3x3s1p2, look at (-2, -2) MPSCNN always looks at
      // (-floor(kernel_size - 1 / 2), -floor(kernel_size - 1 / 2)) Thus, we just
      // need to match this up.

      // For 3x3s1p1, offset should be (0, 0)
      // For 3x3s1p0, offset should be (1, 1)
      // For 3x3s1p2, offset should be (-1, -1)
      const int mps_offset = kernel / 2;
      const int pt_offset = pad;
      return mps_offset - pt_offset;
        */
}
