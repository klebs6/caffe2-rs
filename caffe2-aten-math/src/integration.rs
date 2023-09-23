crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Integration.cpp]

/**
  | The estimated integral of a function y of x,
  |
  | sampled at points (y_1, ..., y_n) that are
  | separated by distance (dx_1, ..., dx_{n-1}),
  |
  | is given by the trapezoid rule:
  |
  | \sum_{i=1}^{n-1}  dx_i * (y_i + y_{i+1}) / 2
  |
  | TODO: if we extend TensorIterator to accept
  | 3 inputs, we can probably make this a bit more
  | performant.
  |
  */
pub fn do_trapz_a(
        y:   &Tensor,
        dx:  &Tensor,
        dim: i64) -> Tensor {
    
    todo!();
        /*
            Tensor left = y.slice(dim, 0, -1);
        Tensor right = y.slice(dim, 1);

        return ((left + right) * dx).sum(dim) / 2.;
        */
}

/**
  | When dx is constant, the above formula
  | simplifies to dx * [(\sum_{i=1}^n y_i)
  | - (y_1 + y_n)/2]
  |
  */
pub fn do_trapz_b(
        y:   &Tensor,
        dx:  f64,
        dim: i64) -> Tensor {
    
    todo!();
        /*
            return (y.sum(dim) - (y.select(dim, 0) + y.select(dim, -1)) * (0.5)) * dx;
        */
}

pub fn zeros_like_except(
        y:   &Tensor,
        dim: i64) -> Tensor {
    
    todo!();
        /*
            auto sizes = y.sizes().vec();
        dim = maybe_wrap_dim(dim, y.dim());
        sizes.erase(sizes.begin() + dim);
        return zeros(sizes, y.options());
        */
}

pub fn trapz_a(
        y:   &Tensor,
        x:   &Tensor,
        dim: i64) -> Tensor {
    
    todo!();
        /*
            dim = maybe_wrap_dim(dim, y);
        // asking for the integral with zero samples is a bit nonsensical,
        // but we'll return "0" to match numpy behavior.
        if (y.size(dim) == 0) {
            return zeros_like_except(y, dim);
        }
        Tensor x_viewed;
        if (x.dim() == 1) {
            TORCH_CHECK(x.size(0) == y.size(dim), "trapz: There must be one `x` value for each sample point");
            DimVector sizes(y.dim(), 1);
            sizes[dim] = x.size(0);
            x_viewed = x.view(sizes);
        } else {
            x_viewed = x;
        }
        Tensor x_left = x_viewed.slice(dim, 0, -1);
        Tensor x_right = x_viewed.slice(dim, 1);

        Tensor dx = x_right - x_left;
        return do_trapz(y, dx, dim);
        */
}

pub fn trapz_b(
        y:   &Tensor,
        dx:  f64,
        dim: i64) -> Tensor {
    
    todo!();
        /*
            // see above
        if (y.size(dim) == 0) {
            return zeros_like_except(y, dim);
        }
        return do_trapz(y, dx, dim);
        */
}
