//! Runtime CPU feature detection for bdver1 capabilities.

/// Returns whether SSE4.2 is available.
#[must_use]
pub fn has_sse4_2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("sse4.2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Returns whether AVX is available.
#[must_use]
pub fn has_avx() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Returns whether AES-NI is available.
#[must_use]
pub fn has_aes() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("aes")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Returns whether POPCNT is available.
#[must_use]
pub fn has_popcnt() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("popcnt")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
