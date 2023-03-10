crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/AffineGridGenerator.cpp]

pub fn linspace_from_neg_one(
        grid:          &Tensor,
        num_steps:     i64,
        align_corners: bool) -> Tensor {
    
    todo!();
        /*
            if (num_steps <= 1) {
        return tensor(0, grid.options());
      }
      auto range = linspace(-1, 1, num_steps, grid.options());
      if (!align_corners) {
        range = range * (num_steps - 1) / num_steps;
      }
      return range;
        */
}

pub fn make_base_grid_4d(
        theta:         &Tensor,
        N:             i64,
        C:             i64,
        H:             i64,
        W:             i64,
        align_corners: bool) -> Tensor {
    
    todo!();
        /*
            auto base_grid = empty({N, H, W, 3}, theta.options());

      base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
      base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
      base_grid.select(-1, 2).fill_(1);

      return base_grid;
        */
}

pub fn make_base_grid_5d(
        theta:         &Tensor,
        N:             i64,
        C:             i64,
        D:             i64,
        H:             i64,
        W:             i64,
        align_corners: bool) -> Tensor {
    
    todo!();
        /*
            auto base_grid = empty({N, D, H, W, 4}, theta.options());

      base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
      base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
      base_grid.select(-1, 2).copy_(linspace_from_neg_one(theta, D, align_corners).unsqueeze_(-1).unsqueeze_(-1));
      base_grid.select(-1, 3).fill_(1);

      return base_grid;
        */
}

pub fn affine_grid_generator_4d(
        theta:         &Tensor,
        N:             i64,
        C:             i64,
        H:             i64,
        W:             i64,
        align_corners: bool) -> Tensor {
    
    todo!();
        /*
            Tensor base_grid = make_base_grid_4D(theta, N, C, H, W, align_corners);
      auto grid = base_grid.view({N, H * W, 3}).bmm(theta.transpose(1, 2));
      return grid.view({N, H, W, 2});
        */
}

pub fn affine_grid_generator_5d(
        theta:         &Tensor,
        N:             i64,
        C:             i64,
        D:             i64,
        H:             i64,
        W:             i64,
        align_corners: bool) -> Tensor {
    
    todo!();
        /*
            Tensor base_grid = make_base_grid_5D(theta, N, C, D, H, W, align_corners);
      auto grid = base_grid.view({N, D * H * W, 4}).bmm(theta.transpose(1, 2));
      return grid.view({N, D, H, W, 3});
        */
}

pub fn affine_grid_generator(
        theta:         &Tensor,
        size:          &[i32],
        align_corners: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          size.size() == 4 || size.size() == 5,
          "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
      if (size.size() == 4) {
        return affine_grid_generator_4D(
            theta, size[0], size[1], size[2], size[3], align_corners);
      } else {
        return affine_grid_generator_5D(
            theta, size[0], size[1], size[2], size[3], size[4], align_corners);
      }
        */
}


pub fn affine_grid_generator_4d_backward(
        grad_grid:     &Tensor,
        N:             i64,
        C:             i64,
        H:             i64,
        W:             i64,
        align_corners: bool) -> Tensor {
    
    todo!();
        /*
            auto base_grid = make_base_grid_4D(grad_grid, N, C, H, W, align_corners);
      AT_ASSERT(grad_grid.sizes() == IntArrayRef({N, H, W, 2}));
      auto grad_theta = base_grid.view({N, H * W, 3})
                            .transpose(1, 2)
                            .bmm(grad_grid.view({N, H * W, 2}));
      return grad_theta.transpose(1, 2);
        */
}

pub fn affine_grid_generator_5d_backward(
        grad_grid:     &Tensor,
        N:             i64,
        C:             i64,
        D:             i64,
        H:             i64,
        W:             i64,
        align_corners: bool) -> Tensor {
    
    todo!();
        /*
            auto base_grid = make_base_grid_5D(grad_grid, N, C, D, H, W, align_corners);
      AT_ASSERT(grad_grid.sizes() == IntArrayRef({N, D, H, W, 3}));
      auto grad_theta = base_grid.view({N, D * H * W, 4})
                            .transpose(1, 2)
                            .bmm(grad_grid.view({N, D * H * W, 3}));
      return grad_theta.transpose(1, 2);
        */
}



pub fn affine_grid_generator_backward(
        grad:          &Tensor,
        size:          &[i32],
        align_corners: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          size.size() == 4 || size.size() == 5,
          "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
      if (size.size() == 4) {
        return affine_grid_generator_4D_backward(
            grad, size[0], size[1], size[2], size[3], align_corners);
      } else {
        return affine_grid_generator_5D_backward(
            grad, size[0], size[1], size[2], size[3], size[4], align_corners);
      }
        */
}
