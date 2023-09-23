crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cuda/BatchLinearAlgebraLib.h]

/// some cusolver functions don't work well on
/// cuda 9.2 or cuda 10.1.105, cusolver is used on
/// cuda >= 10.1.243
///
#[cfg(all(CUDART_VERSION,CUSOLVER_VERSION,CUSOLVER_VERSION_GTE_10200))]
pub const USE_CUSOLVER: bool = true;

// cusolverDn<T>potrfBatched may have numerical
// issue before cuda 11.3 release, (which is
// cusolver version 11101 in the header), so we
// only use cusolver potrf batched if cuda version
// is >= 11.3
//
#[cfg(CUSOLVER_VERSION_GTE_11101)]
pub const USE_CUSOLVER_POTRF_BATCHED: bool = true;

#[cfg(not(CUSOLVER_VERSION_GTE_11101))]
pub const USE_CUSOLVER_POTRF_BATCHED: bool = false;

pub fn geqrf_batched_cublas(
        input: &Tensor,
        tau:   &Tensor)  {
    
    todo!();
        /*
        
        */
}

pub fn triangular_solve_cublas(
        A:                   &mut Tensor,
        B:                   &mut Tensor,
        infos:               &mut Tensor,
        upper:               bool,
        transpose:           bool,
        conjugate_transpose: bool,
        unitriangular:       bool)  {
    
    todo!();
        /*
        
        */
}

pub fn triangular_solve_batched_cublas(
        A:                   &mut Tensor,
        B:                   &mut Tensor,
        infos:               &mut Tensor,
        upper:               bool,
        transpose:           bool,
        conjugate_transpose: bool,
        unitriangular:       bool)  {
    
    todo!();
        /*
        
        */
}

pub fn gels_batched_cublas(
        a:     &Tensor,
        b:     &mut Tensor,
        infos: &mut Tensor)  {
    
    todo!();
        /*
        
        */
}

/**
  | entrance of calculations of `inverse`
  | using cusolver getrf + getrs, cublas
  | getrfBatched + getriBatched
  |
  */
#[cfg(USE_CUSOLVER)]
pub fn inverse_helper_cuda_lib(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
        
        */
}

#[cfg(USE_CUSOLVER)]
pub fn linalg_inv_out_helper_cuda_lib<'a>(
        result:      &mut Tensor,
        infos_getrf: &mut Tensor,
        infos_getrs: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}

/**
  | entrance of calculations of `svd` using
  | cusolver gesvdj and gesvdjBatched
  |
  */
#[cfg(USE_CUSOLVER)]
pub fn svd_helper_cuda_lib(
        self_:      &Tensor,
        some:       bool,
        compute_uv: bool) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
        
        */
}

/**
  | entrance of calculations of `cholesky`
  | using cusolver potrf and potrfBatched
  |
  */
#[cfg(USE_CUSOLVER)]
pub fn cholesky_helper_cusolver(
        input: &Tensor,
        upper: bool,
        info:  &Tensor)  {
    
    todo!();
        /*
        
        */
}

#[cfg(USE_CUSOLVER)]
pub fn cholesky_solve_helper_cuda_cusolver(
        self_: &Tensor,
        A:     &Tensor,
        upper: bool) -> Tensor {
    
    todo!();
        /*
        
        */
}

#[cfg(USE_CUSOLVER)]
pub fn cholesky_inverse_kernel_impl_cusolver<'a>(
        result: &mut Tensor,
        infos:  &mut Tensor,
        upper:  bool) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}

#[cfg(USE_CUSOLVER)]
pub fn geqrf_cusolver(
        input: &Tensor,
        tau:   &Tensor)  {
    
    todo!();
        /*
        
        */
}

#[cfg(USE_CUSOLVER)]
pub fn ormqr_cusolver(
        input:     &Tensor,
        tau:       &Tensor,
        other:     &Tensor,
        left:      bool,
        transpose: bool)  {
    
    todo!();
        /*
        
        */
}

#[cfg(USE_CUSOLVER)]
pub fn orgqr_helper_cusolver<'a>(
        result: &mut Tensor,
        tau:    &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
        
        */
}

#[cfg(USE_CUSOLVER)]
pub fn linalg_eigh_cusolver(
        eigenvalues:          &mut Tensor,
        eigenvectors:         &mut Tensor,
        infos:                &mut Tensor,
        upper:                bool,
        compute_eigenvectors: bool)  {
    
    todo!();
        /*
        
        */
}
