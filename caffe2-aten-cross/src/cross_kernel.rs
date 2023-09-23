crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/CrossKernel.cpp]

pub fn apply_cross<Scalar>(
        result: &mut Tensor,
        a:      &Tensor,
        b:      &Tensor,
        dim:    i64)  {

    todo!();
        /*
            i64 total = a.numel() / 3;
      i64 a_stride = a.stride(dim);
      i64 b_stride = b.stride(dim);
      i64 r_stride = result.stride(dim);

      Scalar *a_ptr = a.data_ptr<Scalar>();
      Scalar *b_ptr = b.data_ptr<Scalar>();
      Scalar *r_ptr = result.data_ptr<Scalar>();

      parallel_for(0, total, internal::GRAIN_SIZE, [&](i64 s, i64 e) {
        const i64 a_dim = a.dim();
        vector<i64> position_in_dims(a_dim);
        i64 index_in_curr_dim = s;
        i64 a_start = 0;
        i64 b_start = 0;
        i64 r_start = 0;
        for (i64 i = 0; i < a.dim(); i++) {
          if (i == dim) continue;
          position_in_dims[i] = index_in_curr_dim % a.size(i);
          a_start += (index_in_curr_dim % a.size(i)) * a.stride(i);
          b_start += (index_in_curr_dim % b.size(i)) * b.stride(i);
          r_start += (index_in_curr_dim % result.size(i)) * result.stride(i);
          index_in_curr_dim = index_in_curr_dim / a.size(i);
        }

        while (s < e) {
          r_ptr[r_start+0*r_stride] = a_ptr[a_start+1*a_stride]*b_ptr[b_start+2*b_stride] - a_ptr[a_start+2*a_stride]*b_ptr[b_start+1*b_stride];
          r_ptr[r_start+1*r_stride] = a_ptr[a_start+2*a_stride]*b_ptr[b_start+0*b_stride] - a_ptr[a_start+0*a_stride]*b_ptr[b_start+2*b_stride];
          r_ptr[r_start+2*r_stride] = a_ptr[a_start+0*a_stride]*b_ptr[b_start+1*b_stride] - a_ptr[a_start+1*a_stride]*b_ptr[b_start+0*b_stride];
          s++;

          for (int i = 0; i < a.dim(); i++) {
            if (i == dim) {
              continue;
            }
            position_in_dims[i]++;
            a_start += a.stride(i);
            b_start += b.stride(i);
            r_start += result.stride(i);
            if (position_in_dims[i] == a.size(i) && i != a.dim()-1) {
                a_start -= position_in_dims[i] * a.stride(i);
                b_start -= position_in_dims[i] * b.stride(i);
                r_start -= position_in_dims[i] * result.stride(i);
                position_in_dims[i] = 0;
            } else {
              break;
            }
          }
        }
      });
        */
}

pub fn cross_kernel_impl(
        result: &mut Tensor,
        a:      &Tensor,
        b:      &Tensor,
        dim:    i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX(result.scalar_type(), "cross", [&]() {
        apply_cross<Scalar>(result, a, b, dim);
      });
        */
}

register_dispatch!{cross_stub, &cross_kernel_impl}
