pub fn cfg_aliases() {
    use cfg_aliases::cfg_aliases;

    // Setup cfg aliases
    cfg_aliases! {
        // Platforms
        wasm:        { target_arch = "wasm32" },
        android:     { target_os = "android" },
        macos:       { target_os = "macos" },
        linux:       { target_os = "linux" },
        not_windows: { not(target_family = "windows") },
    }

    cfg_aliases! {
        i386: { target_arch = "i386" },
        i386_but_not_windows: { all(not(target_family = "windows"),target_arch = "i386") },
        x86_64_or_amd64: {
            any( 
                target_arch = "x64_64", 
                target_arch = "amd64" 
            )
        },
        x86_64_or_amd64_or_i386: {
            any( 
                target_arch = "x64_64", 
                target_arch = "amd64",
                target_arch = "i386" 
            )
        }
    }

    cfg_aliases! {
        have_getcpuid: {
            any(
                target_arch = "x86_64",
                target_arch = "amd64",
                target_arch = "i386"
            )
        }
    }

    cfg_aliases! {
        c2_available: {
            any(
                expose_c2_ops,
                all(
                    not(caffe2_is_xplat_build),
                    not(c10_mobile)
                )
            )
        }
    }
}
