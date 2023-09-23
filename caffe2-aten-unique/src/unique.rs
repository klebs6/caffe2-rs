// # vim: ft=none
/*!
  | Returns unique elements of input tensor.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Unique.cpp]

pub fn unique_cpu_template<Scalar>(
        self_:          &Tensor,
        sorted:         bool,
        return_inverse: bool,
        return_counts:  bool) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            const Tensor& input = self.contiguous();
      const Scalar* input_data = input.data_ptr<Scalar>();
      i64 numel = input.numel();
      Tensor output;
      Tensor inverse_indices = empty({0}, self.options().dtype(kLong));
      Tensor counts = empty({0}, self.options().dtype(kLong));

      unordered_set<Scalar> set(input_data, input_data + numel);
      output = empty({static_cast<i64>(set.size())}, input.options());
      Scalar *output_data = output.data_ptr<Scalar>();

      if (sorted) {
        vector<Scalar> vec(set.begin(), set.end());
        sort(vec.begin(), vec.end());
        copy(vec.begin(), vec.end(), output_data);
      } else {
        copy(set.begin(), set.end(), output_data);
      }

      if (return_inverse || return_counts) {
        inverse_indices.resize_(input.sizes());
        i64* inverse_indices_data = inverse_indices.data_ptr<i64>();
        unordered_map<Scalar, i64> inverse_map;
        inverse_map.reserve(output.numel());
        for (i64 i = 0; i < output.numel(); ++i) {
          inverse_map[output_data[i]] = i;
        }
        for(i64 i = 0; i < numel; ++i) {
          inverse_indices_data[i] = inverse_map[input_data[i]];
        }
        if (return_counts) {
          unordered_map<Scalar, i64> counts_map;
          counts_map.reserve(output.numel());
          for (i64 i = 0; i < output.numel(); ++i) {
            counts_map[output_data[i]] = 0;
          }
          for(i64 i = 0; i < numel; i++) {
            counts_map[input_data[i]] += 1;
          }
          counts.resize_(output.sizes());
          counts.fill_(0);
          i64 *counts_data = counts.data_ptr<i64>();
          for(i64 i = 0; i < output.numel(); i++) {
            counts_data[i] = counts_map[output_data[i]];
          }
        }
      }
      return make_tuple(output, inverse_indices, counts);
        */
}

pub fn unique_consecutive_cpu_template<Scalar>(
    self_:          &Tensor,
    return_inverse: bool,
    return_counts:  bool) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            const Tensor& input = self.contiguous();
      const Scalar* input_data = input.data_ptr<Scalar>();
      i64 numel = input.numel();
      Tensor output = empty({numel}, input.options());
      Tensor inverse_indices = empty({0}, self.options().dtype(kLong));
      Tensor counts = empty({0}, self.options().dtype(kLong));

      if (return_inverse) {
        inverse_indices.resize_(input.sizes());
      }

      if (numel > 0) {
        Scalar *output_data = output.data_ptr<Scalar>();
        i64 *inverse_data = inverse_indices.data_ptr<i64>();;
        i64 *counts_data = nullptr;
        *output_data = *input_data;

        if (return_counts) {
          counts.resize_({numel});
          counts_data = counts.data_ptr<i64>();
        }
        Scalar *p = output_data;
        i64 *q = counts_data;
        i64 last = 0;
        for (i64 i = 0; i < numel; i++) {
          if (input_data[i] != *p) {
            *(++p) = input_data[i];
            if (return_counts) {
              *(q++) = i - last;
              last = i;
            }
          }
          if (return_inverse) {
            inverse_data[i] = p - output_data;
          }
        }
        i64 output_size = p - output_data + 1;
        if (return_counts) {
          *q = numel - last;
          counts.resize_({output_size});
        }
        output.resize_({output_size});
      }

      return make_tuple(output, inverse_indices, counts);
        */
}

pub fn unique_dim_cpu_impl<ForwardIt>(
    first:               ForwardIt,
    last:                ForwardIt,
    indices:             &mut Vec<i64>,
    inverse_indices_vec: Tensor,
    counts:              Tensor) -> ForwardIt {

    todo!();
    /*
            if (first == last) {
          return last;
        }
        // save to calculate distance to iterators
        ForwardIt begin = first;

        // set first inverse index and count
        inverse_indices_vec[indices[0]] = 0;
        counts[0] += 1;

        ForwardIt result = first;
        while (++first != last) {
          if (!equal(*result, *first) && ++result != first) {
              *result = move(*first);
          }
          i64 idx_result = distance(begin, result);
          i64 idx_first = distance(begin, first);
          inverse_indices_vec[indices[idx_first]] = idx_result;
          counts[idx_result] += 1;
        }

        return ++result;
        */
}

pub fn unique_dim_cpu_template<Scalar>(
    self_:          &Tensor,
    dim:            i64,
    consecutive:    bool,
    return_inverse: bool,
    return_counts:  bool) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            auto sizes = self.sizes().vec();
        // check how many zero dimensions exist
        auto num_zero_dims = count(sizes.begin(), sizes.end(), 0);

        // tensor is not well formed as it has 0 sized dimensions
        if (self.size(dim) == 0){
          TORCH_CHECK(
              num_zero_dims == 1,
              "Number of zero sized dimensions is more than one, so unique cannot be applied ")
          Tensor output = empty({0}, self.options());
          Tensor inverse_indices =
              empty({0}, self.options().dtype(kLong));
          Tensor counts = empty({0}, self.options().dtype(kLong));

          return make_tuple(output, inverse_indices, counts);
        }

        TORCH_CHECK(num_zero_dims == 0,
        "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");

      // reshape tensor as [dim, -1]
      Tensor input_flat = self.transpose(dim, 0);
      auto orig_sizes = input_flat.sizes().vec();
      input_flat = input_flat.contiguous().view({input_flat.size(0), -1});

      vector<i64> indices(input_flat.size(0));
      iota(indices.begin(), indices.end(), 0);
      i64 numel = input_flat.size(1);
      Scalar* input_flat_ptr = ((Scalar*)input_flat.data_ptr());

      // sort indices using data
      if (!consecutive) {
        sort(indices.begin(), indices.end(),
          [&](i64 a, i64 b) -> bool {
            for (i64 i = 0; i < numel; ++i) {
              Scalar lhs = input_flat_ptr[i + a * numel];
              Scalar rhs = input_flat_ptr[i + b * numel];
              if (lhs < rhs) {
                return true;
              } else if (lhs > rhs) {
                return false;
              }
            }
            return false;
          });
      }

      Tensor input_sorted;
      if (!consecutive) {
        input_sorted = empty(input_flat.sizes(), input_flat.options());
        for (usize i = 0; i < indices.size(); ++i) {
          input_sorted[i] = input_flat[indices[i]];
        }
      } else {
        input_sorted = input_flat;
      }

      Tensor inverse_indices = empty(indices.size(), self.options().dtype(kLong));
      Tensor counts = zeros(indices.size(), self.options().dtype(kLong));
      vector<Tensor> input_unbind = unbind(input_sorted, 0);
      auto last = _unique_dim_cpu_impl(
        input_unbind.begin(), input_unbind.end(), indices, inverse_indices, counts);
      input_unbind.erase(last, input_unbind.end());
      counts = narrow(counts, 0, 0, input_unbind.size());

      // reshape back
      auto output = stack(input_unbind, 0);
      auto new_sizes = vector<i64>(orig_sizes);
      new_sizes[0] = -1;
      output = output.view(new_sizes);
      output = output.transpose(0, dim);

      return make_tuple(output, inverse_indices, counts);
        */
}

pub fn unique_cpu(
        self_:          &Tensor,
        sorted:         bool,
        return_inverse: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "unique", [&] {
        Tensor output, inverse;
        tie(output, inverse, ignore) = unique_cpu_template<Scalar>(self, sorted, return_inverse, false);
        return make_tuple(output, inverse);
      });
        */
}

pub fn unique2_cpu(
        self_:          &Tensor,
        sorted:         bool,
        return_inverse: bool,
        return_counts:  bool) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            return AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "unique", [&] {
        return unique_cpu_template<Scalar>(self, sorted, return_inverse, return_counts);
      });
        */
}


pub fn unique_dim_cpu(
        self_:          &Tensor,
        dim:            i64,
        sorted:         bool,
        return_inverse: bool,
        return_counts:  bool) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            return AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "unique_dim", [&] {
        // The current implementation using `dim` always sorts due to unhashable tensors
        return _unique_dim_cpu_template<Scalar>(self, dim, false, return_inverse, return_counts);
      });
        */
}


pub fn unique_dim_consecutive_cpu(
        self_:          &Tensor,
        dim:            i64,
        return_inverse: bool,
        return_counts:  bool) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            return AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "unique_dim", [&] {
        return _unique_dim_cpu_template<Scalar>(self, dim, true, return_inverse, return_counts);
      });
        */
}


pub fn unique_consecutive_cpu(
        self_:          &Tensor,
        return_inverse: bool,
        return_counts:  bool,
        dim:            Option<i64>) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            if (!dim.has_value()) {
        return AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, self.scalar_type(), "unique", [&] {
          return unique_consecutive_cpu_template<Scalar>(self, return_inverse, return_counts);
        });
      }
      return unique_dim_consecutive_cpu(self, dim.value(), return_inverse, return_counts);
        */
}
