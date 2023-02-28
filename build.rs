fn main() {

    let lib_dir = "/usr/local/opt/llvm/lib/";

    //println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_dir);
    println!("cargo:rustc-env=LD_LIBRARY_PATH={}", lib_dir);

    //TODO
    //#[cfg(not(cudnn_version_min_gt_6_0_0))]
    //#[cfg(cuda_version_gte_10000)]
}
