use crate::GgmlType;

/// Check if a GGML type is an imatrix quantization type.
///
/// Imatrix quantizations use importance matrices to improve accuracy
/// at lower bit widths by allocating more bits to important weights.
///
/// # Returns
///
/// `true` if the type is an imatrix quantization (IQ1_S, IQ1_M, IQ2_S, IQ2_XS, IQ3_S, IQ3_M, IQ3_XS, IQ4_NL, IQ4_XS),
/// `false` otherwise.
#[inline]
pub fn is_imatrix(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Iq1S
            | GgmlType::Iq1M
            | GgmlType::Iq2S
            | GgmlType::Iq2Xs
            | GgmlType::Iq2Xxs
            | GgmlType::Iq3S
            | GgmlType::Iq3M
            | GgmlType::Iq3Xs
            | GgmlType::Iq3Xxs
            | GgmlType::Iq4Nl
            | GgmlType::Iq4Xs
    )
}

/// Get the description of an imatrix quantization type.
///
/// # Returns
///
/// A descriptive string for the imatrix type, or None if the type is not an imatrix quantization.
#[must_use]
pub fn imatrix_description(ty: GgmlType) -> Option<&'static str> {
    match ty {
        GgmlType::Iq1S => Some("IQ1 S"),
        GgmlType::Iq1M => Some("IQ1 M"),
        GgmlType::Iq2S => Some("IQ2 S"),
        GgmlType::Iq2Xs => Some("IQ2 XS"),
        GgmlType::Iq2Xxs => Some("IQ2 XXS"),
        GgmlType::Iq3S => Some("IQ3 S"),
        GgmlType::Iq3M => Some("IQ3 M"),
        GgmlType::Iq3Xs => Some("IQ3 XS"),
        GgmlType::Iq3Xxs => Some("IQ3 XXS"),
        GgmlType::Iq4Nl => Some("IQ4 NL"),
        GgmlType::Iq4Xs => Some("IQ4 XS"),
        _ => None,
    }
}

/// Get the bit width of an imatrix quantization type.
///
/// # Returns
///
/// The nominal bit width for the imatrix type, or None if the type is not an imatrix quantization.
#[must_use]
pub fn imatrix_bit_width(ty: GgmlType) -> Option<u8> {
    match ty {
        GgmlType::Iq1S | GgmlType::Iq1M => Some(1),
        GgmlType::Iq2S | GgmlType::Iq2Xs | GgmlType::Iq2Xxs => Some(2),
        GgmlType::Iq3S | GgmlType::Iq3M | GgmlType::Iq3Xs | GgmlType::Iq3Xxs => Some(3),
        GgmlType::Iq4Nl | GgmlType::Iq4Xs => Some(4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_imatrix() {
        assert!(is_imatrix(GgmlType::Iq1S));
        assert!(is_imatrix(GgmlType::Iq1M));
        assert!(is_imatrix(GgmlType::Iq2S));
        assert!(is_imatrix(GgmlType::Iq2Xs));
        assert!(is_imatrix(GgmlType::Iq2Xxs));
        assert!(is_imatrix(GgmlType::Iq3S));
        assert!(is_imatrix(GgmlType::Iq3M));
        assert!(is_imatrix(GgmlType::Iq3Xs));
        assert!(is_imatrix(GgmlType::Iq3Xxs));
        assert!(is_imatrix(GgmlType::Iq4Nl));
        assert!(is_imatrix(GgmlType::Iq4Xs));

        assert!(!is_imatrix(GgmlType::F32));
        assert!(!is_imatrix(GgmlType::F16));
        assert!(!is_imatrix(GgmlType::Q4_0));
        assert!(!is_imatrix(GgmlType::Q8_0));
    }

    #[test]
    fn test_imatrix_description() {
        assert_eq!(imatrix_description(GgmlType::Iq1S), Some("IQ1 S"));
        assert_eq!(imatrix_description(GgmlType::Iq1M), Some("IQ1 M"));
        assert_eq!(imatrix_description(GgmlType::Iq2S), Some("IQ2 S"));
        assert_eq!(imatrix_description(GgmlType::Iq2Xs), Some("IQ2 XS"));
        assert_eq!(imatrix_description(GgmlType::Iq2Xxs), Some("IQ2 XXS"));
        assert_eq!(imatrix_description(GgmlType::Iq3S), Some("IQ3 S"));
        assert_eq!(imatrix_description(GgmlType::Iq3M), Some("IQ3 M"));
        assert_eq!(imatrix_description(GgmlType::Iq3Xs), Some("IQ3 XS"));
        assert_eq!(imatrix_description(GgmlType::Iq3Xxs), Some("IQ3 XXS"));
        assert_eq!(imatrix_description(GgmlType::Iq4Nl), Some("IQ4 NL"));
        assert_eq!(imatrix_description(GgmlType::Iq4Xs), Some("IQ4 XS"));

        assert_eq!(imatrix_description(GgmlType::F32), None);
        assert_eq!(imatrix_description(GgmlType::Q4_0), None);
    }

    #[test]
    fn test_imatrix_bit_width() {
        assert_eq!(imatrix_bit_width(GgmlType::Iq1S), Some(1));
        assert_eq!(imatrix_bit_width(GgmlType::Iq1M), Some(1));
        assert_eq!(imatrix_bit_width(GgmlType::Iq2S), Some(2));
        assert_eq!(imatrix_bit_width(GgmlType::Iq2Xs), Some(2));
        assert_eq!(imatrix_bit_width(GgmlType::Iq2Xxs), Some(2));
        assert_eq!(imatrix_bit_width(GgmlType::Iq3S), Some(3));
        assert_eq!(imatrix_bit_width(GgmlType::Iq3M), Some(3));
        assert_eq!(imatrix_bit_width(GgmlType::Iq3Xs), Some(3));
        assert_eq!(imatrix_bit_width(GgmlType::Iq3Xxs), Some(3));
        assert_eq!(imatrix_bit_width(GgmlType::Iq4Nl), Some(4));
        assert_eq!(imatrix_bit_width(GgmlType::Iq4Xs), Some(4));

        assert_eq!(imatrix_bit_width(GgmlType::F32), None);
        assert_eq!(imatrix_bit_width(GgmlType::Q4_0), None);
    }
}
