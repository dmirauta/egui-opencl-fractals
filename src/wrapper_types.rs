use egui_inspect::{EguiInspect, InspectNumber};
use ocl::OclPrm;

#[repr(C)]
#[derive(EguiInspect, Clone, Copy, Debug, Default, PartialEq)]
pub struct BBox {
    #[inspect(min=-2.0, max=2.0)]
    pub left: f64,
    #[inspect(min=-2.0, max=2.0)]
    pub right: f64,
    #[inspect(min=-2.0, max=2.0)]
    pub bot: f64,
    #[inspect(min=-2.0, max=2.0)]
    pub top: f64,
}

unsafe impl OclPrm for BBox {}

#[repr(C)]
#[derive(Debug, EguiInspect, PartialEq, Clone, Copy)]
pub struct Complex {
    #[inspect(min=-2.0, max=2.0)]
    pub re: f64,
    #[inspect(min=-2.0, max=2.0)]
    pub im: f64,
}

impl Default for Complex {
    fn default() -> Self {
        Self { re: -0.7, im: 0.3 }
    }
}

/// Shared fractal params
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SFParam {
    pub mode_int: i32,
    pub c: Complex,
    pub view: BBox,
    pub max_iter: i32,
}

unsafe impl OclPrm for SFParam {}

#[repr(C)]
#[derive(Debug, EguiInspect, PartialEq, Clone, Copy)]
pub struct Freqs {
    #[inspect(log_slider, min = 1.0, max = 1000.0)]
    pub r: f64,
    #[inspect(log_slider, min = 1.0, max = 1000.0)]
    pub g: f64,
    #[inspect(log_slider, min = 1.0, max = 1000.0)]
    pub b: f64,
}

impl Default for Freqs {
    fn default() -> Self {
        Self {
            r: 1.1,
            g: 3.3,
            b: 9.9,
        }
    }
}

unsafe impl OclPrm for Freqs {}

#[repr(C)]
#[derive(Debug, EguiInspect, PartialEq, Clone, Copy)]
pub struct ProxType {
    pub to_unit_circ: bool,
    pub to_horizontal: bool,
    pub to_vertical: bool,
}

impl Default for ProxType {
    fn default() -> Self {
        Self {
            to_unit_circ: true,
            to_horizontal: false,
            to_vertical: false,
        }
    }
}

unsafe impl OclPrm for ProxType {}

#[repr(C)]
#[derive(Debug, Default, EguiInspect, PartialEq, Clone, Copy)]
pub struct ImDims {
    pub height: i32,
    pub width: i32,
}

unsafe impl OclPrm for ImDims {}
