crate::ix!();

#[inline] pub fn rand_int(a: i32, b: i32) -> i32 {
    
    todo!();
    /*
        static std::random_device rd;
      static std::mt19937 gen(rd());

      return std::uniform_int_distribution<int>(a, b)(gen);
    */
}

// TODO(#14383029) cblas_sgemm not yet implemented on limited mobile cases.
#[cfg(target_feature = "neon")]
#[test] fn ConvTransposeMobile_Test() {
    todo!();
    /*
      for (int i = 0; i < 10; ++i) {
        int n = randInt(1, 3);
        int planesIn = randInt(1, 10);
        int h = randInt(10, 200);
        int w = randInt(10, 200);
        int planesOut = randInt(1, 10);
        int kernelH = randInt(2, 5);
        int kernelW = randInt(2, 5);
        int strideH = randInt(1, 4);
        int strideW = randInt(1, 4);
        int padT = randInt(0, 3);
        int padB = randInt(0, 3);
        int padL = 0;
        int padR = 0;
        int adjH = randInt(0, 3);
        if (adjH >= strideH) { adjH = strideH - 1; }
        int adjW = randInt(0, 3);
        if (adjW >= strideW) { adjW = strideW - 1; }

        caffe2::compare(n, planesIn, h, w,
                        planesOut,
                        kernelH, kernelW,
                        strideH, strideW,
                        padT, padL, padB, padR,
                        adjH, adjW, 0.002f, 0.001f);
      }
  */
}
