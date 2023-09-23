crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/LossMulti.h]

pub fn multilabel_margin_loss_shape_check(
        nframe:     &mut i64,
        dim:        &mut i64,
        ndims:      &i64,
        target_arg: &mut TensorArg,
        input:      &Tensor,
        target:     &Tensor)  {
    
    todo!();
        /*
            bool valid_inputs = (ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0;
        TORCH_CHECK(
                    valid_inputs,
                    "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
                    input.sizes());

        if (ndims <= 1) {
          nframe = 1;
          dim = ndims == 0 ? 1 : input.size(0);
          TORCH_CHECK(
                      valid_inputs && target.dim() <= 1 && target.numel() == dim,
                      "inconsistent size ",
                      target.sizes(),
                      " for ",
                      target_arg);
        } else {
          nframe = input.size(0);
          dim = input.size(1);
          TORCH_CHECK(
                      valid_inputs && target.dim() == 2 && target.size(0) == nframe &&
                      target.size(1) == dim,
                      "inconsistent size ",
                      target.sizes(),
                      " for ",
                      target_arg);
        }
        */
}

pub fn multi_margin_loss_shape_check(
        nframe:     &mut i64,
        dim:        &mut i64,
        ndims:      &i64,
        target_arg: &mut TensorArg,
        input:      &Tensor,
        target:     &Tensor)  {
    
    todo!();
        /*
            bool valid_inputs = (ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0;
        if (ndims <= 1) {
          nframe = 1;
          dim = ndims == 0 ? 1 : input.size(0);
        } else {
          nframe = input.size(0);
          dim = input.size(1);
        }

        TORCH_CHECK(
                    valid_inputs,
                    "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
                    input.sizes());
        TORCH_CHECK(
                    valid_inputs && target.dim() <= 1 && target.numel() == nframe,
                    "inconsistent target size, got: ",
                    target.sizes());
        */
}
