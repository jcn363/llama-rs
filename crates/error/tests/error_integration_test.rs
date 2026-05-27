/// Integration tests for the `error` crate.
///
/// These tests verify that the `Error` type works correctly in cross-crate scenarios
/// and that the `Result<T>` alias is usable from external code.
use error::{Error, Result};

#[test]
fn integration_error_can_be_created_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
    let err: Error = io_err.into();
    let msg = format!("{err}");
    assert!(msg.contains("IO error"), "should format as IO error");
    assert!(
        msg.contains("missing file"),
        "should preserve source message"
    );
}

#[test]
fn integration_error_display_is_consistent() {
    let cases: Vec<(Error, &str)> = vec![
        (
            Error::Config("bad config".into()),
            "Configuration error: bad config",
        ),
        (
            Error::Gguf("bad gguf".into()),
            "GGUF parsing error: bad gguf",
        ),
        (Error::Other("generic".into()), "Other error: generic"),
    ];
    for (err, expected) in cases {
        assert_eq!(format!("{err}"), expected);
    }
}

#[test]
fn integration_result_alias_works_in_cross_crate_context() {
    fn fallible_fn() -> Result<i32> {
        Ok(99)
    }
    assert_eq!(fallible_fn().unwrap(), 99);

    fn failing_fn() -> Result<i32> {
        Err(Error::Other("fail".into()))
    }
    assert!(failing_fn().is_err());
}
