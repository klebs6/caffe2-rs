crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ vandermonde_matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn vander(
        x:          &Tensor,
        N:          Option<i64>,
        increasing: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(x.dim() == 1, "x must be a one-dimensional tensor.");

      // Acquires n, defaulting to size if not provided
      i64 n = x.size(0);
      if (N.has_value()) {
        n = *N;
        TORCH_CHECK(n >= 0, "N must be non-negative.");
      }

      // Note: result is long if x is an integer tensor (like int8) because
      // cumprod promotes integer tensors to long
      auto result = empty({x.size(0), n}, x.options().dtype(promote_types(x.scalar_type(), ScalarType::Long)));

      if (n > 0) {
        result.select(1, 0).fill_(1);
      }
      if (n > 1) {
        result.slice(1, 1).copy_(x.unsqueeze(1));
        result.slice(1, 1).copy_(cumprod(result.slice(1, 1), 1));
      }

      if (!increasing) {
        return flip(result, {1});
      }
      return result;
        */
}
