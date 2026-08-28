use feathertalk_domain::{
    CancelFrame, ClientFrame, DomainError, MAX_FRAME_BYTES, PROTOCOL_VERSION, RejectedFrame,
    ServerFrame, ShutdownFrame, StartFrame, TaskId, check_protocol_version, decode_line,
    encode_line,
};

fn task_id() -> TaskId {
    TaskId::parse("1787900000000-0000000a").unwrap()
}

#[test]
fn every_client_frame_round_trips_on_one_line() {
    use feathertalk_domain::{ProbeMediaParams, Request};

    let frames = vec![
        ClientFrame::Start(StartFrame {
            protocol_version: PROTOCOL_VERSION,
            task_id: task_id(),
            request: Request::ProbeMedia(ProbeMediaParams {
                input: std::path::PathBuf::from("a.mov"),
            }),
        }),
        ClientFrame::Cancel(CancelFrame {
            protocol_version: PROTOCOL_VERSION,
            task_id: task_id(),
        }),
        ClientFrame::Shutdown(ShutdownFrame {
            protocol_version: PROTOCOL_VERSION,
        }),
    ];
    for frame in frames {
        let line = encode_line(&frame).unwrap();
        assert!(!line.contains('\n'), "encoded frame contains a newline");
        assert_eq!(decode_line::<ClientFrame>(&line).unwrap(), frame);
    }
}

#[test]
fn a_multiline_string_payload_still_encodes_to_one_line() {
    let frame = ServerFrame::Rejected(RejectedFrame {
        protocol_version: PROTOCOL_VERSION,
        reason: "line one\nline two".to_owned(),
    });
    let line = encode_line(&frame).unwrap();
    assert!(!line.contains('\n'));
    assert!(line.contains(r"line one\nline two"));
    assert_eq!(decode_line::<ServerFrame>(&line).unwrap(), frame);
}

#[test]
fn encoding_an_oversized_frame_is_refused() {
    let frame = ServerFrame::Rejected(RejectedFrame {
        protocol_version: PROTOCOL_VERSION,
        reason: "x".repeat(MAX_FRAME_BYTES),
    });
    assert!(matches!(
        encode_line(&frame),
        Err(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES
        })
    ));
}

#[test]
fn decoding_an_oversized_line_is_refused_before_parsing() {
    let line = format!(r#"{{"frame":"{}"}}"#, "x".repeat(MAX_FRAME_BYTES));
    assert!(matches!(
        decode_line::<ServerFrame>(&line),
        Err(DomainError::FrameTooLong { .. })
    ));
}

#[test]
fn malformed_and_unknown_frames_are_refused() {
    for bad in [
        "",
        "   ",
        "not json",
        r#"{"frame":"greetings","data":{}}"#,
        r#"{"frame":"shutdown","data":{"protocol_version":1,"extra":true}}"#,
    ] {
        assert!(
            matches!(
                decode_line::<ClientFrame>(bad),
                Err(DomainError::MalformedFrame { .. })
            ),
            "expected rejection for {bad:?}"
        );
    }
}

#[test]
fn protocol_version_comparison_is_exact() {
    check_protocol_version(PROTOCOL_VERSION).unwrap();
    for wrong in [0, PROTOCOL_VERSION + 1, u32::MAX] {
        assert!(matches!(
            check_protocol_version(wrong),
            Err(DomainError::ProtocolVersion {
                expected: PROTOCOL_VERSION,
                ..
            })
        ));
    }
}
