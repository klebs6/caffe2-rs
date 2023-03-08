crate::ix!();

#[macro_export] macro_rules! eigen_pow {
    ($x:ident, $y:ident) => {
        todo!();
        /*
                (x.pow(y))
        */
    }
}

struct EigenPowFunctor { }

impl EigenPowFunctor {
    
    #[inline] pub fn run<const b_is_scalar: i32, T1, T2, R>(&mut self, 
        n:       usize,
        a:       *const T1,
        b:       *const T2,
        e:       T2,
        out:     *mut R,
        context: *mut CPUContext)  {
    
        todo!();
        /*
            if (b == NULL) {
          EigenVectorArrayMap<R>(out, n) =
              EIGEN_POW((ConstEigenVectorArrayMap<T1>(a, n)), (e));
        } else {
          if (b_is_scalar) {
            if (b[0] == -1.) {
              EigenVectorArrayMap<R>(out, n) =
                  ConstEigenVectorArrayMap<T1>(a, n).inverse();
            } else if (b[0] == 0.5) {
              EigenVectorArrayMap<R>(out, n) =
                  ConstEigenVectorArrayMap<T1>(a, n).sqrt();
            } else if (b[0] == -0.5) {
              EigenVectorArrayMap<R>(out, n) =
                  ConstEigenVectorArrayMap<T1>(a, n).rsqrt();
            } else if (b[0] == 2.) {
              EigenVectorArrayMap<R>(out, n) =
                  ConstEigenVectorArrayMap<T1>(a, n).square();
            } else {
              EigenVectorArrayMap<R>(out, n) =
                  EIGEN_POW((ConstEigenVectorArrayMap<T1>(a, n)), (b[0]));
            }
          } else {
            EigenVectorArrayMap<R>(out, n) = EIGEN_POW(
                (ConstEigenVectorArrayMap<T1>(a, n)),
                (ConstEigenVectorArrayMap<T2>(b, n)));
          }
        }
        */
    }
    
    #[inline] pub fn run_with_broadcast<T1, T2, R>(&mut self, 
        a:       *const T1,
        b:       *const T2,
        out:     *mut R,
        pre:     usize,
        n:       usize,
        context: *mut CPUContext)  {
    
        todo!();
        /*
            EigenArrayMap<R>(out, n, pre) = EIGEN_POW(
            (ConstEigenArrayMap<T1>(a, n, pre)),
            (ConstEigenVectorArrayMap<T2>(b, n)).rowwise().replicate(pre));
        /*
        //below code only allows elementary ops, such as +, -, * and /,
        //and does not allow operations, such as pow, exp and log
        EIGEN_POW(
           (ConstEigenArrayMap<T>(a, n, pre).colwise()),
           (ConstEigenVectorArrayMap<T>(b, n)));
         */
        */
    }
    
    #[inline] pub fn run_with_broadcast2<T1, T2, R>(&mut self, 
        a:       *const T1,
        b:       *const T2,
        out:     *mut R,
        pre:     usize,
        n:       usize,
        post:    usize,
        context: *mut CPUContext)  {
    
        todo!();
        /*
            for (auto i = 0U; i < pre; ++i) {
          EigenArrayMap<R>(out + i * n * post, post, n) = EIGEN_POW(
              (ConstEigenArrayMap<T1>(a + i * n * post, post, n)),
              (Eigen::Map<const Eigen::Array<T2, 1, Eigen::Dynamic>>(b, n))
                  .colwise()
                  .replicate(post));
          /*
          //below code only allows elementary ops, such as +, -, * and /,
          //and does not allow for operations, such as pow, exp and log
          EIEGN_POW(
            (ConstEigenArrayMap<T>(a + i * n * post, post, n).rowwise()),
            (Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>>(b, n)));
          */
        }
        */
    }
}
