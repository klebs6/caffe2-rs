crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/WeightNorm.cpp]

/**
  | Staying faithful to the Python for now
  | for clarity, look for optimizations
  | later (e.g., single return statement
  | for RVO)
  |
  */
pub fn norm_except_dim(
        v:   &Tensor,
        pow: i64,
        dim: i64) -> Tensor {
    
    todo!();
        /*
            // I assume tensor.contiguous(), view(), norm(), etc. here will dispatch through VariableType.
      if (dim == -1) {
        return v.norm(pow);
      } else if (dim == 0) {
        vector<i64> output_size(v.dim(), 1);
        output_size[0] = v.size(0);
        return v.contiguous().view({v.size(0), -1}).norm(pow, 1).view(output_size);
      } else if (dim == v.dim() - 1) {
        vector<i64> output_size(v.dim(), 1);
        output_size[v.dim() - 1] = v.size(v.dim() - 1);
        return v.contiguous().view({-1, v.size(v.dim() - 1)}).norm(pow, 0).view(output_size);
      } else {
        // To consider: native::norm_except_dim is probably fine as well,
        // and would avoid an additional dynamic dispatch.
        return norm_except_dim(v.transpose(0, dim), pow, 0).transpose(0, dim); // optimize?
      }
        */
}

pub fn weight_norm(
        v_in: &Tensor,
        g_in: &Tensor,
        dim:  i64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
        v_in.device() == g_in.device(),
        "weight_norm: expected v_in and g_in to be on the same device, but v_in is "
        "on ", v_in.device(), " and g_in is on ", g_in.device());

      auto v = v_in.contiguous();
      auto g = g_in.contiguous();

      bool can_use_fused = v.is_cuda() && (dim == 0 || dim == v.dim() - 1);

      if (can_use_fused) {
        // weight_norm does not have a derivative defined for it, so this will route back through
        // VariableType.cpp, and construct a WeightNormFusedBackward object in the autograd graph.
        return get<0>(_weight_norm_cuda_interface(v, g, dim));
      } else {
        // Double-differentiable primitive ops
        // native::norm_except_dim would probably be fine as well.
        return v*(g/norm_except_dim(v, 2, dim));
      }
        */
}

/**
  | Differentiable backward path, an alternative
  | to weight_norm_cuda_backward, to
  | be used when backward is itself creating
  | a graph.
  | 
  | The GradMode::is_enabled() check
  | must be performed within Functions.cpp;
  | that's why we define a separate function
  | here, instead of inlining it in weight_norm_cuda_backward.
  |
  */
pub fn weight_norm_differentiable_backward(
        grad_w:      &Tensor,
        saved_v:     &Tensor,
        saved_g:     &Tensor,
        saved_norms: &Tensor,
        dim:         i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            // In Functions.cpp, the HardshrinkBackward object supplies "grad.contiguous()"
      // as the first argument, so grad_w should be contiguous here.
      // All these checks should succeed:
      TORCH_CHECK(grad_w.is_contiguous(), "grad_w must be contiguous");
      TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
      TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
      TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");

      i64 last_dim = saved_v.dim() - 1;
      i64 last_size = saved_v.size(last_dim);

      // Like weight_norm_fused_backward, weight_norm_differentiable_backward should only ever be called
      // through a WeightNormFusedBackward object, so we expect that dim == 0 || dim == saved_v.size(-1)
      TORCH_CHECK(dim == 0 || dim == last_dim, "Expected dim to be the first or last dimension");

      // saved_g and saved_norms are already shaped to broadcast over the correct dimensions

      // ...but saved_norms might be Float when saved_g and saved_v are half.
      // To consider:  saved_norms.to(..., True /*non_blocking*/);
      auto norms = saved_norms.to(saved_g.scalar_type());

      vector<i64> bcast_size(saved_v.dim(), 1);

      // Analytic backward path using differentiable primitive ops
      if (dim == 0) {
        bcast_size[0] = saved_v.size(0);
        auto per_dim_sums = (grad_w*saved_v).view({saved_v.size(0), -1}).sum(1).view(bcast_size);
        auto grad_v = (saved_g/norms)*(grad_w - saved_v*(per_dim_sums/(norms*norms)));
        auto grad_g = per_dim_sums/norms;
        return tuple<Tensor, Tensor>{grad_v, grad_g};
      } else { // dim == last_dim
        bcast_size[last_dim] = last_size;
        auto per_dim_sums = (grad_w*saved_v).view({-1, last_size}).sum(0).view(bcast_size);
        auto grad_v = (saved_g/norms)*(grad_w - saved_v*(per_dim_sums/(norms*norms)));
        auto grad_g = per_dim_sums/norms;
        return tuple<Tensor, Tensor>{grad_v, grad_g};
      }
        */
}
