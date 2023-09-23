crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/DistanceOpsKernel.cpp]

pub struct Dist<Scalar> {

}

pub type VectorizedScalar = Vectorized<Scalar>;

pub mod dist {

    use super::*;

    /* ------------------- * Zero norm  ------------------- */
    pub struct ZDistCalc<Data> {

    }

    impl ZDistCalc<Data> {
        
        #[inline] pub fn map(
            diff: &Data,
            p:    &Data) -> Data {
            
            todo!();
            /*
                return min(ceil(abs(diff)), 1);
            */
        }
        
        #[inline] pub fn red(
            agg: &Data,
            up:  &Data) -> Data {
            
            todo!();
            /*
                return agg + up;
            */
        }
        
        #[inline] pub fn finish(
            agg: Scalar,
            p:   Scalar) -> Scalar {
            
            todo!();
            /*
                return agg;
            */
        }
    }

    /* ------------------- * One norm  ------------------- */
    pub struct ODistCalc<Data> {

    }

    impl ODistCalc<Data> {
        
        #[inline] pub fn map(
            diff: &Data,
            p:    &Data) -> Data {
            
            todo!();
            /*
                return diff;
            */
        }
        
        #[inline] pub fn red(
            agg: &Data,
            up:  &Data) -> Data {
            
            todo!();
            /*
                return agg + up;
            */
        }
        
        #[inline] pub fn finish(
            agg: Scalar,
            p:   Scalar) -> Scalar {
            
            todo!();
            /*
                return agg;
            */
        }
        
        #[inline] pub fn backward(
            diff: &VectorizedScalar,
            grad: Scalar,
            dist: Scalar,
            p:    &VectorizedScalar) -> VectorizedScalar {
            
            todo!();
            /*
                return VectorizedScalar(grad) * sign(diff);
            */
        }
    }

    /**
      | Special general pnorm derivative if
      | p is less than two
      |
      */
    pub struct LttDistCalc {

    }

    impl LttDistCalc {
        
        #[inline] pub fn backward(
            diff: &VectorizedScalar,
            grad: Scalar,
            dist: Scalar,
            p:    &VectorizedScalar) -> VectorizedScalar {
            
            todo!();
            /*
                VectorizedScalar result = (dist == 0.0) ? VectorizedScalar(0) : (sign(diff) * diff.abs().pow(p - VectorizedScalar(1)) * VectorizedScalar(grad) / VectorizedScalar(dist).pow(p - VectorizedScalar(1)));
              result = VectorizedScalar::blendv(result, VectorizedScalar(0), (diff == VectorizedScalar(0)) & (p < VectorizedScalar(1)));
              return result;
            */
        }
    }

    // Two norm
    pub struct TDistCalc<Data> {

    }

    impl TDistCalc<Data> {

        /**
          | TODO This can probably use fused add
          | multiply to get better perf
          |
          */
        #[inline] pub fn map(
            diff: &Data,
            p:    &Data) -> Data {
            
            todo!();
            /*
                return diff * diff;
            */
        }
        
        #[inline] pub fn red(
            agg: &Data,
            up:  &Data) -> Data {
            
            todo!();
            /*
                return agg + up;
            */
        }
        
        #[inline] pub fn finish(
            agg: Scalar,
            p:   Scalar) -> Scalar {
            
            todo!();
            /*
                return sqrt(agg);
            */
        }
        
        #[inline] pub fn backward(
            diff: &VectorizedScalar,
            grad: Scalar,
            dist: Scalar,
            p:    &VectorizedScalar) -> VectorizedScalar {
            
            todo!();
            /*
                return dist == 0.0 ? VectorizedScalar(0) : VectorizedScalar(grad) * diff / VectorizedScalar(dist);
            */
        }
    }

    /* ---------------- * General p norm  ---------------- */
    pub struct PDistCalc<Data> {

    }

    impl PDistCalc<Data> {
        
        #[inline] pub fn map(
            diff: &Data,
            p:    &Data) -> Data {
            
            todo!();
            /*
                return pow(diff, p);
            */
        }
        
        #[inline] pub fn red(
            agg: &Data,
            up:  &Data) -> Data {
            
            todo!();
            /*
                return agg + up;
            */
        }
        
        #[inline] pub fn finish(
            agg: Scalar,
            p:   Scalar) -> Scalar {
            
            todo!();
            /*
                return pow(agg, 1.0 / p);
            */
        }
        
        #[inline] pub fn backward(
            diff: &VectorizedScalar,
            grad: Scalar,
            dist: Scalar,
            p:    &VectorizedScalar) -> VectorizedScalar {
            
            todo!();
            /*
                return dist == 0.0 ? VectorizedScalar(0) : diff * diff.abs().pow(p - VectorizedScalar(2)) * VectorizedScalar(grad) / VectorizedScalar(dist).pow(p - VectorizedScalar(1));
            */
        }
    }

    // Inf norm
    //
    pub struct IDistCalc<Data> {

    }

    impl IDistCalc<Data> {
        
        #[inline] pub fn map(
            diff: &Data,
            p:    &Data) -> Data {
            
            todo!();
            /*
                return diff;
            */
        }
        
        #[inline] pub fn red(
            agg: &Data,
            up:  &Data) -> Data {
            
            todo!();
            /*
                return max(agg, up);
            */
        }
        
        #[inline] pub fn finish(
            agg: Scalar,
            p:   Scalar) -> Scalar {
            
            todo!();
            /*
                return agg;
            */
        }

        /**
          | TODO This backward pass uses a very
          | complext expression to compute (diff ==
          | dist) that could be much faster if using
          | SSE instructions.
          */
        #[inline] pub fn backward(
            diff: &VectorizedScalar,
            grad: Scalar,
            dist: Scalar,
            p:    &VectorizedScalar) -> VectorizedScalar {
            
            todo!();
            /*
                return VectorizedScalar(grad) * sign(diff) * (VectorizedScalar(1) - vec::minimum(VectorizedScalar(1), (diff.abs() - VectorizedScalar(dist)).abs().ceil()));
            */
        }
    }
}

impl Dist<Scalar> {

    // Depending on the value of the pnorm, there
    // are specific implementations that are much
    // faster than pow(abs(a - b), p), but have the
    // same standard loop code for how to process
    // the input vector. To reuse the main outside
    // loop while still guaranteeing that the
    // compiler inlines every different function on
    // p, we break the inner norm logic into structs
    // with static functions that represent what's
    // done differently, and template the outer loop
    // on those structs.
    //
    // The four functions are:
    //
    //     map :      This tells how to modify (a - b) 
    //                to form the component that gets summed.
    //
    //     red :      This tells how to sum the
    //                result of map up. This is
    //                separate because the inf norm
    //                actually uses max instead of
    //                sum.
    //
    //     finish :   This tells what to do with the
    //                aggregated value to compute
    //                the norm. Generally this is
    //                the result of val ^ (1 / p).
    //
    //     backward : This is the gradient for that
    //                norm. Arguments are pretty
    //                self explanitory.
    //
    // There are a few cases where these aren't
    // used. The 0 norm has no backward, because
    // it's always 0, so that's shortcircuited
    // earlier. There's a special implementation of
    // the general backward pass when p is less than
    // two, so there's a struct with only a backward
    // pass for this case.
    //

    /**
      | TODO This is an inefficient way to compite
      | sign, and can be much faster using native SSE
      | instructions that should be added to
      | Vectorized.
      */
    #[inline] pub fn sign(val: VectorizedScalar) -> VectorizedScalar {
        
        todo!();
        /*
            return vec::minimum(vec::maximum(VectorizedScalar(0), val.ceil()), VectorizedScalar(1)) +
          vec::minimum(vec::maximum(VectorizedScalar(-1), val.floor()), VectorizedScalar(0));
        */
    }
    
    #[inline] pub fn abs(val: VectorizedScalar) -> VectorizedScalar {
        
        todo!();
        /*
            return val.abs();
        */
    }
    
    #[inline] pub fn abs(val: Scalar) -> Scalar {
        
        todo!();
        /*
            return abs(val);
        */
    }
    
    #[inline] pub fn ceil(val: VectorizedScalar) -> VectorizedScalar {
        
        todo!();
        /*
            return val.ceil();
        */
    }
    
    #[inline] pub fn ceil(val: Scalar) -> Scalar {
        
        todo!();
        /*
            return ceil(val);
        */
    }
    
    #[inline] pub fn min(
        val:   VectorizedScalar,
        other: Scalar) -> VectorizedScalar {
        
        todo!();
        /*
            return vec::minimum(val, VectorizedScalar(other));
        */
    }
    
    #[inline] pub fn min(
        val:   Scalar,
        other: Scalar) -> Scalar {
        
        todo!();
        /*
            return min(val, other);
        */
    }
    
    #[inline] pub fn max(
        val:   VectorizedScalar,
        other: VectorizedScalar) -> VectorizedScalar {
        
        todo!();
        /*
            return vec::maximum(val, other);
        */
    }
    
    #[inline] pub fn max(
        val:   Scalar,
        other: Scalar) -> Scalar {
        
        todo!();
        /*
            return max(val, other);
        */
    }
    
    #[inline] pub fn pow(val: VectorizedScalar, p: VectorizedScalar) -> VectorizedScalar {
        
        todo!();
        /*
            return val.pow(p);
        */
    }
    
    #[inline] pub fn pow(
        val: Scalar,
        p:   Scalar) -> Scalar {
        
        todo!();
        /*
            return pow(val, p);
        */
    }
    
    pub fn run_parallel_pdist<F>(
        result: &mut Tensor,
        self_:  &Tensor,
        p:      Scalar)  {
    
        todo!();
        /*
            const Scalar * const self_start = self.data_ptr<Scalar>();
        const Scalar * const self_end = self_start + self.numel();
        i64 n = self.size(0);
        i64 m = self.size(1);

        Scalar * const res_start = result.data_ptr<Scalar>();
        i64 combs = result.numel(); // n * (n - 1) / 2

        // We conceptually iterate over tuples of (i, j, k) where i is the first
        // vector from the input, j is the second, and k is the result index. This
        // parallelizes over the range of k and infers what i and j are from the
        // value of k.
        parallel_for(0, combs, internal::GRAIN_SIZE / (16 * m), [p, self_start, self_end, n, m, res_start](i64 k, i64 end) {
          const VectorizedScalar pvec(p);
          double n2 = n - .5;
          // The -1 accounts for floating point truncation issues
          i64 i = static_cast<i64>((n2 - sqrt(n2 * n2 - 2 * k - 1)));
          i64 j = k - n * i + i * (i + 1) / 2 + i + 1;

          const Scalar * self_i = self_start + i * m;
          const Scalar * self_j = self_start + j * m;
          Scalar * res = res_start + k;
          const Scalar * const res_end = res_start + end;

          while (res != res_end) {
            *res = F::finish(vec::map2_reduce_all<Scalar>(
              [&pvec](VectorizedScalar a, VectorizedScalar b) { return F::map((a - b).abs(), pvec); },
              F::red, self_i, self_j, m), p);

            res += 1;
            self_j += m;
            if (self_j == self_end) {
              self_i += m;
              self_j = self_i + m;
            }
          }
        });
        */
    }

    /**
      | Assumes self is nonempty, contiguous,
      | and 2D
      |
      */
    pub fn apply_pdist(
        result: &mut Tensor,
        self_:  &Tensor,
        p:      Scalar)  {
        
        todo!();
        /*
            if (p == 0.0) {
          run_parallel_pdist<zdist_calc<VectorizedScalar>>(result, self, p);
        } else if (p == 1.0) {
          run_parallel_pdist<odist_calc<VectorizedScalar>>(result, self, p);
        } else if (p == 2.0) {
          run_parallel_pdist<tdist_calc<VectorizedScalar>>(result, self, p);
        } else if (isinf(p)) {
          run_parallel_pdist<idist_calc<VectorizedScalar>>(result, self, p);
        } else {
          run_parallel_pdist<pdist_calc<VectorizedScalar>>(result, self, p);
        }
        */
    }
    
    pub fn run_parallel_cdist<F>(
        result: &mut Tensor,
        t1:     &Tensor,
        t2:     &Tensor,
        p:      Scalar)  {
    
        todo!();
        /*
            const Scalar * const t1_start = t1.data_ptr<Scalar>();
        const Scalar * const t2_start = t2.data_ptr<Scalar>();
        i64 d = t1.size(0);
        i64 r1 = t1.size(-2);
        i64 r2 = t2.size(-2);
        i64 m = t1.size(-1);

        Scalar * const res_start = result.data_ptr<Scalar>();
        i64 combs = r1 * r2;
        i64 size1 = r1 * m;
        i64 size2 = r2 * m;

        parallel_for(0, combs * d, internal::GRAIN_SIZE / (16 * m), [=](i64 start, i64 end) {
          Scalar * res = res_start + start;
          const Scalar * const res_end = res_start + end;
          i64 l = start / combs;
          i64 k = start % combs;
          i64 i = k / r2;
          i64 j = k % r2;
          i = i * m;
          j = j * m;

          while (res != res_end) {
            const Scalar * self_i = t1_start + size1 * l + i;
            const Scalar * self_j = t2_start + size2 * l + j;

            Scalar agg = 0;
            for (int x = 0; x < m; x++) {
              Scalar a = *(self_i + x);
              Scalar b = *(self_j + x);
              agg = F::red(agg, F::map(abs(a-b), p));
            }
            *res = F::finish(agg, p);

            res += 1;
            j += m;
            if (j == size2) {
              j = 0;
              i += m;
              if (i == size1) {
                i = 0;
                l += 1;
              }
            }
          }
        });
        */
    }
    
    pub fn apply_cdist(
        result: &mut Tensor,
        x1:     &Tensor,
        x2:     &Tensor,
        p:      Scalar)  {
        
        todo!();
        /*
            if (p == 0.0) {
          run_parallel_cdist<zdist_calc<Scalar>>(result, x1, x2, p);
        } else if (p == 1.0) {
          run_parallel_cdist<odist_calc<Scalar>>(result, x1, x2, p);
        } else if (p == 2.0) {
          run_parallel_cdist<tdist_calc<Scalar>>(result, x1, x2, p);
        } else if (isinf(p)) {
          run_parallel_cdist<idist_calc<Scalar>>(result, x1, x2, p);
        } else {
          run_parallel_cdist<pdist_calc<Scalar>>(result, x1, x2, p);
        }
        */
    }

    /**
      | This does a backward pass down a VectorizedScalar column
      | of the input
      |
      */
    #[inline] pub fn backward_down_column_pdist<F>(
        self_i: *const Scalar,
        res_i:  *mut Scalar,
        grad_k: *const Scalar,
        dist_k: *const Scalar,
        pvec:   &VectorizedScalar,
        n:      i64,
        m:      i64,
        gs:     i64,
        count:  i64)  {
    
        todo!();
        /*
            for (const Scalar * const self_end = self_i + m * n; self_i != self_end - m; self_i += m, res_i += m) {

          const VectorizedScalar self_vec_i = VectorizedScalar::loadu(self_i, count);
          VectorizedScalar res_vec_i = VectorizedScalar::loadu(res_i, count);

          const Scalar * self_j = self_i + m;
          Scalar * res_j = res_i + m;
          for (; self_j != self_end; self_j += m, res_j += m, grad_k += gs, dist_k += 1) {
            const VectorizedScalar self_vec_j = VectorizedScalar::loadu(self_j, count);
            VectorizedScalar res_vec_j = VectorizedScalar::loadu(res_j, count);

            VectorizedScalar res = F::backward(self_vec_i - self_vec_j, *grad_k, *dist_k, pvec);
            res_vec_i = res_vec_i + res;
            res_vec_j = res_vec_j - res;

            res_vec_j.store(res_j, count);
          }

          res_vec_i.store(res_i, count);
        }
        */
    }
    
    pub fn run_backward_parallel_pdist<F>(
        result: &mut Tensor,
        grad:   &Tensor,
        self_:  &Tensor,
        p:      Scalar,
        dist:   &Tensor)  {
    
        todo!();
        /*
            const i64 n = self.size(0);
        const i64 m = self.size(1);
        const i64 gs = grad.stride(0);

        const Scalar * const grad_start = grad.data_ptr<Scalar>();
        const Scalar * const dist_start = dist.data_ptr<Scalar>();
        const Scalar * const self_start = self.data_ptr<Scalar>();
        Scalar * const res_start = result.data_ptr<Scalar>();

        // The only way to parallelize and avoid locking requires parallelizing
        // over the columns of the input, i.e. we compute the gradient for the
        // first section of each vector independentaly of the second section, etc.
        parallel_for(0, m / VectorizedScalar::size(), internal::GRAIN_SIZE / (8 * n * n), [p, n, m, gs, grad_start, dist_start, self_start, res_start](i64 l, i64 end) {
          const VectorizedScalar pvec(p);

          const Scalar * self_l = self_start + l * VectorizedScalar::size();
          Scalar * res_l = res_start + l * VectorizedScalar::size();

          for (const Scalar * const res_end = res_start + end * VectorizedScalar::size(); res_l != res_end; self_l += VectorizedScalar::size(), res_l += VectorizedScalar::size()) {
            backward_down_column_pdist<F>(self_l, res_l, grad_start, dist_start, pvec, n, m, gs);
          }
        });
        const i64 remainder = m % VectorizedScalar::size();
        if (remainder) {
          backward_down_column_pdist<F>(self_start + (m - remainder), res_start + (m - remainder), grad_start, dist_start, VectorizedScalar(p), n, m, gs, remainder);
        }
        */
    }

    /**
      | Assumes self is nonempty, contiguous,
      | and 2D and dist is also contiguous
      |
      */
    pub fn apply_backward_pdist(
        result: &mut Tensor,
        grad:   &Tensor,
        self_:  &Tensor,
        p:      f64,
        dist:   &Tensor)  {
        
        todo!();
        /*
            result.fill_(0);
        if (p == 0.0) {
        } else if (p == 1.0) {
          run_backward_parallel_pdist<odist_calc<VectorizedScalar>>(result, grad, self, p, dist);
        } else if (p < 2.0) {
          run_backward_parallel_pdist<lttdist_calc>(result, grad, self, p, dist);
        } else if (p == 2.0) {
          run_backward_parallel_pdist<tdist_calc<VectorizedScalar>>(result, grad, self, p, dist);
        } else if (isinf(p)) {
          run_backward_parallel_pdist<idist_calc<VectorizedScalar>>(result, grad, self, p, dist);
        } else {
          run_backward_parallel_pdist<pdist_calc<VectorizedScalar>>(result, grad, self, p, dist);
        }
        */
    }
    
    pub fn apply_backward_cdist(
        result: &mut Tensor,
        grad:   &Tensor,
        x1:     &Tensor,
        x2:     &Tensor,
        p:      f64,
        dist:   &Tensor)  {
        
        todo!();
        /*
            result.fill_(0);
        if (p == 0.0) {
        } else if (p == 1.0) {
          run_backward_parallel_cdist<odist_calc<VectorizedScalar>>(result, grad, x1, x2, p, dist);
        } else if (p < 2.0) {
          run_backward_parallel_cdist<lttdist_calc>(result, grad, x1, x2, p, dist);
        } else if (p == 2.0) {
          run_backward_parallel_cdist<tdist_calc<VectorizedScalar>>(result, grad, x1, x2, p, dist);
        } else if (isinf(p)) {
          run_backward_parallel_cdist<idist_calc<VectorizedScalar>>(result, grad, x1, x2, p, dist);
        } else {
          run_backward_parallel_cdist<pdist_calc<VectorizedScalar>>(result, grad, x1, x2, p, dist);
        }
        */
    }
    
    pub fn run_backward_parallel_cdist<F>(
        result: &mut Tensor,
        grad:   &Tensor,
        t1:     &Tensor,
        t2:     &Tensor,
        p:      Scalar,
        dist:   &Tensor)  {
    
        todo!();
        /*
            const i64 r1 = t1.size(-2);
        const i64 r2 = t2.size(-2);
        const i64 m = t1.size(-1);
        const i64 d = result.size(0);
        const i64 l1_size = r1 * m;
        const i64 l2_size = r2 * m;
        //current implementation supports only tensor that can be collapsed to 1D. However, to avoid checking if grad satisfies this assumption,
        //we call .contiguous() on grad before backward, thus stride is guaranteed to be 1
        //don't use grad.stride(-1), because if last dimension is 1, stride can be bogus.
        const i64 gs = 1;

        const Scalar * const grad_start = grad.data_ptr<Scalar>();
        const Scalar * const dist_start = dist.data_ptr<Scalar>();
        const Scalar * const t1_start = t1.data_ptr<Scalar>();
        const Scalar * const t2_start = t2.data_ptr<Scalar>();
        Scalar * const res_start = result.data_ptr<Scalar>();

        parallel_for(0, m / VectorizedScalar::size(), internal::GRAIN_SIZE / (16 * r1), [=](i64 l, i64 end) {
          const VectorizedScalar pvec(p);

          const Scalar * i = t1_start + l * VectorizedScalar::size();
          const Scalar * j = t2_start + l * VectorizedScalar::size();
          Scalar * res_l = res_start + l * VectorizedScalar::size();

          for (const Scalar * const res_end = res_start + end * VectorizedScalar::size(); res_l != res_end; i += VectorizedScalar::size(), j += VectorizedScalar::size(), res_l += VectorizedScalar::size()) {
            backward_down_column_cdist<F>(i, j, res_l, grad_start, dist_start, pvec, r1, r2, m, d, gs, l1_size, l2_size);
          }
        });
        const i64 remainder = m % VectorizedScalar::size();
        if (remainder) {
          backward_down_column_cdist<F>(t1_start + (m - remainder), t2_start + (m - remainder), res_start + (m - remainder), grad_start, dist_start, VectorizedScalar(p), r1, r2, m, d, gs, l1_size, l2_size, remainder);
        }
        */
    }
    
    #[inline] pub fn backward_down_column_cdist<F>(
        t1:      *const Scalar,
        t2:      *const Scalar,
        res:     *mut Scalar,
        grad_k:  *const Scalar,
        dist_k:  *const Scalar,
        pvec:    &VectorizedScalar,
        r1:      i64,
        r2:      i64,
        m:       i64,
        d:       i64,
        gs:      i64,
        l1_size: i64,
        l2_size: i64,
        count:   i64)  {
    
        todo!();
        /*
            const Scalar * t1_end = t1 + l1_size;
        const Scalar * t2_end = t2 + l2_size;

        for (i64 l = 0; l < d; l++) {
          for (; t1 != t1_end; t1 += m, res += m) {
            const VectorizedScalar vec_t1 = VectorizedScalar::loadu(t1, count);
            VectorizedScalar res_vec = VectorizedScalar::loadu(res, count);

            for (const Scalar * t2_curr = t2; t2_curr != t2_end; t2_curr += m, grad_k += gs, dist_k += 1) {
              const VectorizedScalar vec_t2 = VectorizedScalar::loadu(t2_curr, count);
              VectorizedScalar res = F::backward(vec_t1 - vec_t2, *grad_k, *dist_k, pvec);
              res_vec = res_vec + res;
            }

            res_vec.store(res, count);
          }
          t1_end += l1_size;
          t2_end += l2_size;
          t2 += l2_size;
        }
        */
    }
}

pub fn pdist_forward_kernel_impl(
        result: &mut Tensor,
        self_:  &Tensor,
        p:      f64)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "pdist", [&] {
        Dist<Scalar>::apply_pdist(result, self, p);
      });
        */
}

pub fn pdist_backward_kernel_impl(
        result: &mut Tensor,
        grad:   &Tensor,
        self_:  &Tensor,
        p:      f64,
        dist:   &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "pdist_backward", [&] {
        Dist<Scalar>::apply_backward_pdist(result, grad, self, p, dist);
      });
        */
}

pub fn cdist_kernel_impl(
        result: &mut Tensor,
        x1:     &Tensor,
        x2:     &Tensor,
        p:      f64)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "cdist", [&] {
        Dist<Scalar>::apply_cdist(result, x1, x2, p);
      });
        */
}

pub fn cdist_backward_kernel_impl(
        result: &mut Tensor,
        grad:   &Tensor,
        x1:     &Tensor,
        x2:     &Tensor,
        p:      f64,
        dist:   &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "cdist_backward", [&] {
        Dist<Scalar>::apply_backward_cdist(result, grad, x1, x2, p, dist);
      });
        */
}

register_dispatch!{pdist_forward_stub  , &pdist_forward_kernel_impl}
register_dispatch!{pdist_backward_stub , &pdist_backward_kernel_impl}
register_dispatch!{cdist_stub          , &cdist_kernel_impl}
register_dispatch!{cdist_backward_stub , &cdist_backward_kernel_impl}
