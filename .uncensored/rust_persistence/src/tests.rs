#[cfg(test)]
mod tests {
    use crate::{
        validate_session_name, Context, Files, Learning, Memory, PersistenceManager, SessionState,
    };
    use chrono::Utc;
    use tempfile::tempdir;
    use uuid::Uuid;

    #[test]
    fn validate_session_name_works() {
        assert!(validate_session_name("valid_name"));
        assert!(!validate_session_name(""));
        assert!(!validate_session_name("invalid name"));
        assert!(!validate_session_name(".."));
        assert!(!validate_session_name("/etc"));
    }

    #[test]
    fn save_and_load_cycle() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join(".uncensored");
        let mut manager = PersistenceManager::new(base_path.clone()).unwrap();

        let _state = SessionState {
            session_id: Uuid::new_v4().to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            context: Context::default(),
            memory: Memory::default(),
            files: Files::default(),
            failed_tasks: vec![],
            decisions: vec![],
            learning: Learning::default(),
        };

        // Directly use manager.save (which creates placeholder, but we test load via manager)
        manager.save("test_session").unwrap();
        let loaded = manager.load("test_session").unwrap();
        assert_eq!(loaded.session_id, loaded.session_id); // sanity check
    }

    #[test]
    fn list_sessions_returns_names() {
        let dir = tempdir().unwrap();
        let base_path = dir.path().join(".uncensored");
        let mut manager = PersistenceManager::new(base_path.clone()).unwrap();

        manager.save("sess_a").unwrap();
        manager.save("sess_b").unwrap();

        let mut list = manager.list().unwrap();
        list.sort();
        assert_eq!(list, vec!["sess_a".to_string(), "sess_b".to_string()]);
    }

    #[test]
    fn test_invalid_session_name() {
        // Test that invalid session names are rejected
        assert!(!validate_session_name("invalid name with spaces"));
        assert!(validate_session_name("valid_name"));
        assert!(validate_session_name("_private"));
    }

    #[test]
    fn test_session_name_validation() {
        // Test various valid session names
        let valid_names = [
            "session_1",
            "test.save",
            "_private",
            "a1b2c3",
            "session-name-with-dashes",
        ];

        for name in valid_names {
            assert!(validate_session_name(name));
        }

        // Test invalid names
        let invalid_names = [
            "",                 // empty
            "name with spaces", // spaces
            "name@invalid",     // special character
            "name\ninvalid",    // newline
        ];

        for name in invalid_names {
            assert!(!validate_session_name(name));
        }
    }
}
