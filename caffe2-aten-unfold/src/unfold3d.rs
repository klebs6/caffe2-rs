crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Unfold3d.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Unfold3d.cpp]

pub fn is_age_zero_and_altb(a: i64, b: i64) -> bool {
    
    todo!();
        /*
            return static_cast<u64>(a) < static_cast<u64>(b);
        */
}

pub fn mat_copy_a<T>(
        M:   i64,
        N:   i64,
        lda: i64,
        ldb: i64,
        A:   *const T,
        B:   *mut T)  {

    todo!();
        /*
            for (i64 i = 0; i < M; ++i) {
        memcpy(B + i * ldb, A + i * lda, N * sizeof(T));
      }
        */
}

pub fn mat_copy_b<T>(
        M:       i64,
        N:       i64,
        lda:     i64,
        stridea: i64,
        ldb:     i64,
        strideb: i64,
        A:       *const T,
        B:       *mut T)  {

    todo!();
        /*
            for (i64 i = 0; i < M; ++i) {
        const T* A_ptr = A + i * lda;
        T* B_ptr = B + i * ldb;
        for (i64 j = 0; j < N; ++j) {
          B_ptr[j * strideb] = A_ptr[j * stridea];
        }
      }
        */
}

// Y += X
pub fn mat_add_a<T>(
        M:   i64,
        N:   i64,
        ldx: i64,
        ldy: i64,
        X:   *const T,
        Y:   *mut T)  {

    todo!();
        /*
            for (i64 i = 0; i < M; ++i) {
        for (i64 j = 0; j < N; ++j) {
          Y[i * ldy + j] += X[i * ldx + j];
        }
      }
        */
}

// Y += X
pub fn mat_add_b<T>(
        M:       i64,
        N:       i64,
        ldx:     i64,
        stridex: i64,
        ldy:     i64,
        stridey: i64,
        X:       *const T,
        Y:       *mut T)  {

    todo!();
        /*
            for (i64 i = 0; i < M; ++i) {
        for (i64 j = 0; j < N; ++j) {
          Y[i * ldy + j * stridey] += X[i * ldx + j * stridex];
        }
      }
        */
}

#[cfg(AT_MKL_ENABLED)]
mod mkl_enabled {
    use super::*;

    pub fn mat_copy_float(
            M:   i64,
            N:   i64,
            lda: i64,
            ldb: i64,
            A:   *const f32,
            B:   *mut f32)  {
        
        todo!();
            /*
                mkl_somatcopy('R', 'N', M, N, 1.0f, A, lda, B, ldb);
            */
    }

    pub fn mat_copy_double(
            M:   i64,
            N:   i64,
            lda: i64,
            ldb: i64,
            A:   *const f64,
            B:   *mut f64)  {
        
        todo!();
            /*
                mkl_domatcopy('R', 'N', M, N, 1.0, A, lda, B, ldb);
            */
    }

    pub fn mat_copy_float(
            M:       i64,
            N:       i64,
            lda:     i64,
            stridea: i64,
            ldb:     i64,
            strideb: i64,
            A:       *const f32,
            B:       *mut f32)  {
        
        todo!();
            /*
                mkl_somatcopy2('R', 'N', M, N, 1.0f, A, lda, stridea, B, ldb, strideb);
            */
    }

    pub fn mat_copy_double(
            M:       i64,
            N:       i64,
            lda:     i64,
            stridea: i64,
            ldb:     i64,
            strideb: i64,
            A:       *const f64,
            B:       *mut f64)  {
        
        todo!();
            /*
                mkl_domatcopy2('R', 'N', M, N, 1.0, A, lda, stridea, B, ldb, strideb);
            */
    }

    pub fn mat_add_float(
            M:   i64,
            N:   i64,
            ldx: i64,
            ldy: i64,
            X:   *const f32,
            Y:   *mut f32)  {
        
        todo!();
            /*
                mkl_somatadd('R', 'N', 'N', M, N, 1.0f, X, ldx, 1.0f, Y, ldy, Y, ldy);
            */
    }

    pub fn mat_add_double(
            M:   i64,
            N:   i64,
            ldx: i64,
            ldy: i64,
            X:   *const f64,
            Y:   *mut f64)  {
        
        todo!();
            /*
                mkl_domatadd('R', 'N', 'N', M, N, 1.0, X, ldx, 1.0, Y, ldy, Y, ldy);
            */
    }

    pub fn mat_add_a(
        M:       i64,
        N:       i64,
        ldx:     i64,
        stridex: i64,
        ldy:     i64,
        stridey: i64,
        X:       *const f32,
        Y:       *mut f32)  {
        
        todo!();
            /*
                for (i64 i = 0; i < M; ++i) {
                cblas_saxpy(N, 1.0f, X + i * ldx, stridex, Y + i * ldy, stridey);
              }
            */
    }

    pub fn mat_add_b(
        M:       i64,
        N:       i64,
        ldx:     i64,
        stridex: i64,
        ldy:     i64,
        stridey: i64,
        X:       *const f64,
        Y:       *mut f64)  {

        todo!();
            /*
                for (i64 i = 0; i < M; ++i) {
                cblas_daxpy(N, 1.0, X + i * ldx, stridex, Y + i * ldy, stridey);
              }
            */
    }
}

pub fn unfold3d_zero_padding_copy_kernel_impl<T>(
        C:        i64,
        X_D:      i64,
        X_H:      i64,
        X_W:      i64,
        Y_D:      i64,
        Y_H:      i64,
        Y_W:      i64,
        kernel_d: i64,
        kernel_h: i64,
        kernel_w: i64,
        stride_d: i64,
        stride_h: i64,
        stride_w: i64,
        src:      *const T,
        dst:      *mut T)  {

    todo!();
        /*
            const i64 n = C * kernel_d * kernel_h * kernel_w;
      const i64 X_size = X_D * X_H * X_W;
      const i64 Y_size = Y_D * Y_H * Y_W;
      parallel_for(0, n, 0, [=](i64 begin, i64 end) {
        for (i64 p = begin; p < end; ++p) {
          i64 c = p;
          const i64 kw = c % kernel_w;
          c /= kernel_w;
          const i64 kh = c % kernel_h;
          c /= kernel_h;
          const i64 kd = c % kernel_d;
          c /= kernel_d;
          for (i64 yd = 0; yd < Y_D; ++yd) {
            const i64 xd = yd * stride_d + kd;
            const T* src_ptr = src + c * X_size + xd * X_H * X_W + kh * X_W + kw;
            T* dst_ptr = dst + p * Y_size + yd * Y_H * Y_W;
            if (stride_w == 1) {
              MatCopy<T>(Y_H, Y_W, stride_h * X_W, Y_W, src_ptr, dst_ptr);
            } else {
              MatCopy<T>(
                  Y_H, Y_W, stride_h * X_W, stride_w, Y_W, 1, src_ptr, dst_ptr);
            }
          }
        }
      });
        */
}

pub fn unfold3d_copy_kernel_impl<T>(
        C:        i64,
        X_D:      i64,
        X_H:      i64,
        X_W:      i64,
        Y_D:      i64,
        Y_H:      i64,
        Y_W:      i64,
        kernel_d: i64,
        kernel_h: i64,
        kernel_w: i64,
        stride_d: i64,
        stride_h: i64,
        stride_w: i64,
        pad_d:    i64,
        pad_h:    i64,
        pad_w:    i64,
        src:      *const T,
        dst:      *mut T)  {

    todo!();
        /*
            if (pad_d == 0 && pad_h == 0 && pad_w == 0) {
        Unfold3dZeroPaddingCopyKernelImpl<T>(
            C,
            X_D,
            X_H,
            X_W,
            Y_D,
            Y_H,
            Y_W,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            src,
            dst);
        return;
      }

      const i64 n = C * kernel_d * kernel_h * kernel_w;
      const i64 X_size = X_D * X_H * X_W;
      const i64 Y_size = Y_D * Y_H * Y_W;
      parallel_for(0, n, 0, [=](i64 begin, i64 end) {
        for (i64 p = begin; p < end; ++p) {
          i64 c = p;
          const i64 kw = c % kernel_w;
          c /= kernel_w;
          const i64 kh = c % kernel_h;
          c /= kernel_h;
          const i64 kd = c % kernel_d;
          c /= kernel_d;
          const T* src_ptr = src + c * X_size;
          T* dst_ptr = dst + p * Y_size;
          for (i64 yd = 0; yd < Y_D; ++yd) {
            const i64 xd = yd * stride_d - pad_d + kd;
            if (!IsAGeZeroAndALtB(xd, X_D)) {
              memset(dst_ptr + yd * Y_H * Y_W, 0, Y_H * Y_W * sizeof(T));
              continue;
            }
            for (i64 yh = 0; yh < Y_H; ++yh) {
              const i64 xh = yh * stride_h - pad_h + kh;
              if (!IsAGeZeroAndALtB(xh, X_H)) {
                memset(
                    dst_ptr + yd * Y_H * Y_W + yh * Y_W, 0, Y_W * sizeof(T));
                continue;
              }
              for (i64 yw = 0; yw < Y_W; ++yw) {
                const i64 xw = yw * stride_w - pad_w + kw;
                dst_ptr[yd * Y_H * Y_W + yh * Y_W + yw] = IsAGeZeroAndALtB(xw, X_W)
                    ? src_ptr[xd * X_H * X_W + xh * X_W + xw]
                    : T(0);
              }
            }
          }
        }
      });
        */
}

pub fn unfold3d_zero_padding_acc_kernel_impl<T>(
        C:        i64,
        X_D:      i64,
        X_H:      i64,
        X_W:      i64,
        Y_D:      i64,
        Y_H:      i64,
        Y_W:      i64,
        kernel_d: i64,
        kernel_h: i64,
        kernel_w: i64,
        stride_d: i64,
        stride_h: i64,
        stride_w: i64,
        src:      *const T,
        dst:      *mut T)  {

    todo!();
        /*
            const i64 X_size = X_D * X_H * X_W;
      const i64 Y_size = Y_D * Y_H * Y_W;
      const i64 kernel_size = kernel_d * kernel_h * kernel_w;
      parallel_for(0, C, 0, [=](i64 begin, i64 end) {
        memset(dst + begin * X_size, 0, (end - begin) * X_size * sizeof(T));
        for (i64 c = begin; c < end; ++c) {
          for (i64 kd = 0; kd < kernel_d; ++kd) {
            for (i64 kh = 0; kh < kernel_h; ++kh) {
              for (i64 kw = 0; kw < kernel_w; ++kw) {
                const i64 p =
                    c * kernel_size + kd * kernel_h * kernel_w + kh * kernel_w + kw;
                for (i64 yd = 0; yd < Y_D; ++yd) {
                  const i64 xd = yd * stride_d + kd;
                  const T* src_ptr = src + p * Y_size + yd * Y_H * Y_W;
                  T* dst_ptr = dst + c * X_size + xd * X_H * X_W + kh * X_W + kw;
                  if (stride_w == 1) {
                    MatAdd<T>(Y_H, Y_W, Y_W, stride_h * X_W, src_ptr, dst_ptr);
                  } else {
                    MatAdd<T>(
                        Y_H,
                        Y_W,
                        Y_W,
                        1,
                        stride_h * X_W,
                        stride_w,
                        src_ptr,
                        dst_ptr);
                  }
                }
              }
            }
          }
        }
      });
        */
}

pub fn unfold3d_acc_kernel_impl<T>(
        C:        i64,
        X_D:      i64,
        X_H:      i64,
        X_W:      i64,
        Y_D:      i64,
        Y_H:      i64,
        Y_W:      i64,
        kernel_d: i64,
        kernel_h: i64,
        kernel_w: i64,
        stride_d: i64,
        stride_h: i64,
        stride_w: i64,
        pad_d:    i64,
        pad_h:    i64,
        pad_w:    i64,
        src:      *const T,
        dst:      *mut T)  {

    todo!();
        /*
            if (pad_d == 0 && pad_h == 0 && pad_w == 0) {
        Unfold3dZeroPaddingAccKernelImpl<T>(
            C,
            X_D,
            X_H,
            X_W,
            Y_D,
            Y_H,
            Y_W,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            src,
            dst);
        return;
      }
      const i64 X_size = X_D * X_H * X_W;
      const i64 Y_size = Y_D * Y_H * Y_W;
      const i64 kernel_size = kernel_d * kernel_h * kernel_w;
      parallel_for(0, C, 0, [=](i64 begin, i64 end) {
        memset(dst + begin * X_size, 0, (end - begin) * X_size * sizeof(T));
        for (i64 c = begin; c < end; ++c) {
          T* dst_ptr = dst + c * X_size;
          for (i64 kd = 0; kd < kernel_d; ++kd) {
            for (i64 kh = 0; kh < kernel_h; ++kh) {
              for (i64 kw = 0; kw < kernel_w; ++kw) {
                const i64 p =
                    c * kernel_size + kd * kernel_h * kernel_w + kh * kernel_w + kw;
                const T* src_ptr = src + p * Y_size;
                for (i64 yd = 0; yd < Y_D; ++yd) {
                  const i64 xd = yd * stride_d - pad_d + kd;
                  if (!IsAGeZeroAndALtB(xd, X_D)) {
                    continue;
                  }
                  for (i64 yh = 0; yh < Y_H; ++yh) {
                    const i64 xh = yh * stride_h - pad_h + kh;
                    if (!IsAGeZeroAndALtB(xh, X_H)) {
                      continue;
                    }
                    for (i64 yw = 0; yw < Y_W; ++yw) {
                      const i64 xw = yw * stride_w - pad_w + kw;
                      if (IsAGeZeroAndALtB(xw, X_W)) {
                        dst_ptr[xd * X_H * X_W + xh * X_W + xw] +=
                            src_ptr[yd * Y_H * Y_W + yh * Y_W + yw];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      });
        */
}

pub fn unfold3d_copy_cpu(
        src:      &Tensor,
        C:        i64,
        X_D:      i64,
        X_H:      i64,
        X_W:      i64,
        Y_D:      i64,
        Y_H:      i64,
        Y_W:      i64,
        kernel_d: i64,
        kernel_h: i64,
        kernel_w: i64,
        stride_d: i64,
        stride_h: i64,
        stride_w: i64,
        pad_d:    i64,
        pad_h:    i64,
        pad_w:    i64,
        dst:      *mut Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(
          ScalarType::BFloat16,
          src.scalar_type(),
          "Unfold3dCopyCPU",
          [=, &src]() {
            Unfold3dCopyKernelImpl<Scalar>(
                C,
                X_D,
                X_H,
                X_W,
                Y_D,
                Y_H,
                Y_W,
                kernel_d,
                kernel_h,
                kernel_w,
                stride_d,
                stride_h,
                stride_w,
                pad_d,
                pad_h,
                pad_w,
                src.data_ptr<Scalar>(),
                dst->data_ptr<Scalar>());
          });
        */
}

pub fn unfold3d_acc_cpu(
        src:      &Tensor,
        C:        i64,
        X_D:      i64,
        X_H:      i64,
        X_W:      i64,
        Y_D:      i64,
        Y_H:      i64,
        Y_W:      i64,
        kernel_d: i64,
        kernel_h: i64,
        kernel_w: i64,
        stride_d: i64,
        stride_h: i64,
        stride_w: i64,
        pad_d:    i64,
        pad_h:    i64,
        pad_w:    i64,
        dst:      *mut Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(
          ScalarType::BFloat16,
          src.scalar_type(),
          "Unfold3dAccCPU",
          [=, &src]() {
            Unfold3dAccKernelImpl<Scalar>(
                C,
                X_D,
                X_H,
                X_W,
                Y_D,
                Y_H,
                Y_W,
                kernel_d,
                kernel_h,
                kernel_w,
                stride_d,
                stride_h,
                stride_w,
                pad_d,
                pad_h,
                pad_w,
                src.data_ptr<Scalar>(),
                dst->data_ptr<Scalar>());
          });
        */
}
