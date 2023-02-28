crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Dropout.cpp]

pub fn make_feature_noise(input: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto input_sizes = input.sizes();
      TORCH_CHECK(input.dim() >= 2, "Feature dropout requires at least 2 dimensions in the input");
      vector<i64> sizes;
      sizes.reserve(input.dim());
      sizes.push_back(input_sizes[0]);
      sizes.push_back(input_sizes[1]);
      for (i64 i = 2; i < input.dim(); ++i)
        sizes.push_back(1);
      return empty(sizes, input.options());
        */
}

pub fn is_fused_kernel_acceptable(
        input: &Tensor,
        p:     f64) -> bool {
    
    todo!();
        /*
            return input.is_cuda() && p > 0 && p < 1 && input.numel() > 0;
        */
}

/**
  | NB: sure, we could have used different
  | overloads here, but I would feel insecure
  | knowing that this dispatch depends only on the
  | constness of the references
  |
  */
pub fn multiply_a<const inplace: bool>(
        input: &mut Tensor,
        noise: &Tensor) -> &mut Tensor {

    todo!();
        /*
            static_assert(inplace, "Wrong multiply overload triggered in Dropout.cpp");
      return input.mul_(noise);
        */
}

pub fn multiply_b<const inplace: bool>(
        input: &Tensor,
        noise: &Tensor) -> Tensor {

    todo!();
        /*
            static_assert(!inplace, "Wrong multiply overload triggered in Dropout.cpp");
      return input.mul(noise);
        */
}

pub fn dropout_impl<const feature_dropout: bool, const alpha_dropout: bool, T>(
    input: &mut T,
    p:     f64,
    train: bool) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
      if (p == 0 || !train || input.numel() == 0) {
        return input;
      }

      if (p == 1) {
        return multiply<inplace>(input, zeros({}, input.options()));
      }

      Tensor b; // used for alpha_dropout only
      auto noise = feature_dropout ? make_feature_noise(input) : empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      noise.bernoulli_(1 - p);
      if (alpha_dropout) {
        constexpr double alpha = 1.7580993408473766;
        double a = 1. / sqrt((alpha * alpha * p + 1) * (1 - p));
        b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);
        noise.mul_(a);
      } else {
        noise.div_(1 - p);
      }

      if (!alpha_dropout) {
        return multiply<inplace>(input, noise);
      } else {
        return multiply<inplace>(input, noise).add_(b);
      }
        */
}

//NOTE: in the rust, it doesn't look like we ought to express anything like this because it is an
//anti pattern.  we will just have to deal with returning `Tensor` until it becomes clear we need
//another way.
//
//pub type Ctype<const inplace: bool> = Conditional<inplace, Tensor&, Tensor>::type;

#[macro_export] macro_rules! alias_specialization {
    ($ALIAS_NAME:ident, $IS_FEATURE:ident, $IS_ALPHA:ident) => {
        /*
        
        template <bool inplace, typename... Args>                                           
        Ctype<inplace> ALIAS_NAME(Args&&... args) {                                         
          return _dropout_impl<IS_FEATURE, IS_ALPHA, inplace>(forward<Args>(args)...); 
        }
        */
    }
}

alias_specialization!{_dropout,               false, false}
alias_specialization!{_feature_dropout,       true,  false}
alias_specialization!{_alpha_dropout,         false, true }
alias_specialization!{_feature_alpha_dropout, true,  true }

pub fn dropout_a(
        input: &Tensor,
        p:     f64,
        train: bool) -> Tensor {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        if (train && is_fused_kernel_acceptable(input, p)) {
          return get<0>(_fused_dropout(input, 1 - p));
        }
        return _dropout<false>(input, p, train);
      }();
      namedinference::propagate_names(result, input);
      return result;
        */
}

pub fn dropout_b(
        input: &mut Tensor,
        p:     f64,
        train: bool) -> &mut Tensor {
    
    todo!();
        /*
            return _dropout<true>(input, p, train);
        */
}

pub fn feature_dropout_a(
        input: &Tensor,
        p:     f64,
        train: bool) -> Tensor {
    
    todo!();
        /*
            return _feature_dropout<false>(input, p, train);
        */
}

pub fn feature_dropout_b(
        input: &mut Tensor,
        p:     f64,
        train: bool) -> &mut Tensor {
    
    todo!();
        /*
            return _feature_dropout<true>(input, p, train);
        */
}

pub fn alpha_dropout_a(
        input: &Tensor,
        p:     f64,
        train: bool) -> Tensor {
    
    todo!();
        /*
            return _alpha_dropout<false>(input, p, train);
        */
}

pub fn alpha_dropout_b(
        input: &mut Tensor,
        p:     f64,
        train: bool) -> &mut Tensor {
    
    todo!();
        /*
            return _alpha_dropout<true>(input, p, train);
        */
}

pub fn feature_alpha_dropout_a(
        input: &Tensor,
        p:     f64,
        train: bool) -> Tensor {
    
    todo!();
        /*
            return _feature_alpha_dropout<false>(input, p, train);
        */
}

pub fn feature_alpha_dropout_b(
        input: &mut Tensor,
        p:     f64,
        train: bool) -> &mut Tensor {
    
    todo!();
        /*
            return _feature_alpha_dropout<true>(input, p, train);
        */
}
