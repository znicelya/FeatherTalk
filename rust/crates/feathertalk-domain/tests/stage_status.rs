use feathertalk_domain::{ErrorCode, TaskStage, TaskStatus};

#[test]
fn the_stage_vocabulary_has_thirteen_variants() {
    assert_eq!(TaskStage::ALL_UNIT_SAMPLES.len(), 13);
}

#[test]
fn every_stage_projects_to_exactly_one_status() {
    for stage in TaskStage::ALL_UNIT_SAMPLES {
        let expected = match &stage {
            TaskStage::Queued => TaskStatus::Queued,
            TaskStage::Completed => TaskStatus::Completed,
            TaskStage::Failed { .. } => TaskStatus::Failed,
            TaskStage::Cancelled => TaskStatus::Cancelled,
            _ => TaskStatus::Running,
        };
        assert_eq!(stage.status(), expected, "{stage:?}");
    }
}

#[test]
fn only_completed_failed_and_cancelled_are_terminal() {
    for stage in TaskStage::ALL_UNIT_SAMPLES {
        let expected = matches!(
            stage,
            TaskStage::Completed | TaskStage::Failed { .. } | TaskStage::Cancelled
        );
        assert_eq!(stage.is_terminal(), expected, "{stage:?}");
    }
}

#[test]
fn data_carrying_stages_use_adjacent_tagging_on_the_wire() {
    let training = TaskStage::Training {
        epoch: 3,
        step: 1200,
        loss: 0.0425,
    };
    assert_eq!(
        serde_json::to_string(&training).unwrap(),
        r#"{"stage":"training","data":{"epoch":3,"step":1200,"loss":0.0425}}"#
    );
    assert_eq!(
        serde_json::to_string(&TaskStage::Preparing).unwrap(),
        r#"{"stage":"preparing"}"#
    );
    let failed = TaskStage::Failed {
        code: ErrorCode::DiskSpaceLow,
        message: "磁盘空间不足".to_owned(),
    };
    let json = serde_json::to_string(&failed).unwrap();
    assert_eq!(serde_json::from_str::<TaskStage>(&json).unwrap(), failed);
}

#[test]
fn task_error_stage_must_not_be_terminal() {
    use feathertalk_domain::{DomainError, TaskError};

    let ok = TaskError::new(
        ErrorCode::DiskSpaceLow,
        "磁盘空间不足",
        "needed 4 GiB",
        TaskStage::Rendering {
            frame: 10,
            total: 900,
        },
    );
    ok.validate().unwrap();

    let bad = TaskError::new(
        ErrorCode::DiskSpaceLow,
        "磁盘空间不足",
        "needed 4 GiB",
        TaskStage::Completed,
    );
    assert!(matches!(
        bad.validate(),
        Err(DomainError::InvalidField { field: "stage", .. })
    ));
}
