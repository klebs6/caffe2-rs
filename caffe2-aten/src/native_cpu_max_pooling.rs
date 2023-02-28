crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/MaxPooling.cpp]

#[inline] pub fn max_pool1d_kernel<Scalar>(
        op: *mut Scalar,
        ip: *const Scalar,
        p:  &PoolingParams1D)  {

    todo!();
        /*
            for (i64 kj = 0; kj < p.KW; ++kj) {
        i64 oj = p.valid_output_start(kj);
        i64 oe = p.valid_output_end(kj);
        i64 ij = p.index(kj, oj);
        for (; oj < oe; ++oj, ij += p.SJ) {
          Scalar val = ip[ij];
          bool update_max = isnan(val) || op[oj] < val;
          op[oj] = update_max ? val : op[oj];
        }
      }
        */
}

pub fn max_pool1d_impl(
        output: &mut Tensor,
        input:  &Tensor,
        p:      &PoolingParams1D)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool1d_impl", [&] {
        const Tensor in = input.contiguous();
        Scalar* const OP = output.data_ptr<Scalar>();
        const Scalar* const IP = in.data_ptr<Scalar>();

        // Value used for padding
        constexpr Scalar FILL = numeric_limits<Scalar>::has_infinity
            ? -numeric_limits<Scalar>::infinity()
            : numeric_limits<Scalar>::lowest();

        parallel_for(0, p.NB * p.NC, 0, [&](i64 begin, i64 end) {
          for (i64 it = begin; it < end; ++it) {
            Scalar* op = OP + it * p.OW;
            const Scalar* ip = IP + it * p.IW;
            fill_n(op, p.OW, FILL);
            max_pool1d_kernel(op, ip, p);
          }
        });
      });
        */
}

register_dispatch!{max_pool1d_stub, &max_pool1d_impl}
