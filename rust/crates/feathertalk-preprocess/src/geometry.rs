use crate::{Landmarks, PreprocessError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceBoundingBox {
    pub xmin: i32,
    pub ymin: i32,
    pub xmax: i32,
    pub ymax: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropSpec {
    pub crop_size: u32,
    pub inner_size: u32,
    pub border: u32,
    pub mouth_mask: MaskRect,
}

pub fn compute_face_bbox(landmarks: &Landmarks) -> Result<FaceBoundingBox, PreprocessError> {
    let points = landmarks.points();
    let xmin = points[1].x as i32;
    let ymin = points[52].y as i32;
    let xmax = points[31].x as i32;
    let width = xmax - xmin;
    if width <= 0 {
        return Err(PreprocessError::InvalidGeometry {
            field: "face_bbox",
            message: "xmax must be greater than xmin".into(),
        });
    }
    let ymax = ymin
        .checked_add(width)
        .ok_or_else(|| PreprocessError::InvalidGeometry {
            field: "face_bbox",
            message: "ymax overflow".into(),
        })?;
    Ok(FaceBoundingBox {
        xmin,
        ymin,
        xmax,
        ymax,
    })
}

pub fn default_crop_spec() -> CropSpec {
    CropSpec {
        crop_size: 168,
        inner_size: 160,
        border: 4,
        mouth_mask: MaskRect {
            x: 5,
            y: 5,
            width: 150,
            height: 145,
        },
    }
}
