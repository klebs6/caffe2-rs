crate::ix!();

#[macro_export] macro_rules! pairwise_diff {
    ($vec:ident, $N:ident) => {
        /*
      ((vec.matrix() * Eigen::MatrixXf::Ones(1, N) -            
        Eigen::MatrixXf::Ones(N, 1) * vec.matrix().transpose()) 
           .array())
      */
      }
}

#[macro_export] macro_rules! cwise_sigm {
    ($vec:ident) => {
        /*
           (1. / (1. + (-(vec)).exp()))
        */
      }
}

#[macro_export] macro_rules! cwise_gt {
    ($vec1:ident, $vec2:ident) => {
        /*
           ((vec1) > (vec2))
        */
      }
}

#[macro_export] macro_rules! cwise_lt {
    ($vec1:ident, $vec2:ident) => {
        /*
           ((vec1) < (vec2))
        */
      }
}

#[macro_export] macro_rules! cwise_sign {
    ($vec:ident) => {
        /*
           (CWISE_GT((vec), 0).cast<float>() * 2. - 1.)
        */
      }
}

#[macro_export] macro_rules! cwise_log_sigm {
    ($vec:ident, $huge:ident) => {
        /*
      (CWISE_GT((vec), (huge))        
           .select(                   
               0, CWISE_LT((vec), -(huge)).select(vec, CWISE_SIGM((vec)).log())))
      */
      }
}
