crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cudnn/AffineGridGenerator.cpp]

// See Note [ATen preprocessor philosophy]
#[cfg(not(AT_CUDNN_ENABLED))]
pub fn cudnn_affine_grid_generator_forward(
        theta: &Tensor,
        n:     i64,
        c:     i64,
        h:     i64,
        w:     i64) -> Tensor {
    
    todo!();
        /*
            AT_ERROR("cudnn_affine_grid_generator_forward: ATen not compiled with cuDNN support");
        */
}

#[cfg(not(AT_CUDNN_ENABLED))]
pub fn cudnn_affine_grid_generator_backward(
        grad_theta: &Tensor,
        n:          i64,
        c:          i64,
        h:          i64,
        w:          i64) -> Tensor {
    
    todo!();
        /*
            AT_ERROR("cudnn_affine_grid_generator_backward: ATen not compiled with cuDNN support");
        */
}

#[cfg(AT_CUDNN_ENABLED)]
pub fn set_sampler_descriptor(
        desc:      &mut SpatialTransformerDescriptor,
        data_type: CudnnDataType,
        N:         i32,
        C:         i32,
        H:         i32,
        W:         i32)  {
    
    todo!();
        /*
            int inputSize[4] = {N, C, H, W};
      desc.set(dataType, 4, inputSize);
        */
}

#[cfg(AT_CUDNN_ENABLED)]
pub fn cudnn_affine_grid_generator_forward(
        theta_t: &Tensor,
        N:       i64,
        C:       i64,
        H:       i64,
        W:       i64) -> Tensor {
    
    todo!();
        /*
            auto theta_t_contig = theta_t.contiguous();
      TensorArg theta{ theta_t_contig, "theta", 1 };
      CheckedFrom c = "cudnn_affine_grid_generator_forward";
      checkContiguous(c, theta);
      checkSize(c, theta, {N, 2, 3});

      auto grid_t = empty({0}, theta->options());
      grid_t.resize_({N, H, W, 2});

      auto dataType = getCudnnDataType(*theta);
      SpatialTransformerDescriptor desc;
      setSamplerDescriptor(desc, dataType, N, C, H, W);
      AT_CUDNN_CHECK(cudnnSpatialTfGridGeneratorForward(getCudnnHandle(), desc.desc(),
                                                     theta->data_ptr(),
                                                     grid_t.data_ptr()));
      return grid_t;
        */
}

#[cfg(AT_CUDNN_ENABLED)]
pub fn cudnn_affine_grid_generator_backward(
        grad_grid_t: &Tensor,
        n:           i64,
        c:           i64,
        h:           i64,
        w:           i64) -> Tensor {
    
    todo!();
        /*
            auto grad_grid_contig = grad_grid_t.contiguous();
      TensorArg grad_grid{ grad_grid_contig, "grad_grid", 1 };
      CheckedFrom c = "cudnn_affine_grid_generator_backward";
      checkContiguous(c, grad_grid);
      checkSize(c, grad_grid, {N, H, W, 2});

      auto grad_theta_t = empty({0}, grad_grid->options());
      grad_theta_t.resize_({N, 2, 3});

      auto dataType = getCudnnDataType(grad_theta_t);
      SpatialTransformerDescriptor desc;
      setSamplerDescriptor(desc, dataType, N, C, H, W);
      AT_CUDNN_CHECK(cudnnSpatialTfGridGeneratorBackward(getCudnnHandle(), desc.desc(),
                                                      grad_grid->data_ptr(),
                                                      grad_theta_t.data_ptr()));
      return grad_theta_t;
        */
}
