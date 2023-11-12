extern crate ocl;
use eframe::NativeOptions;
use egui::RichText;
use egui_inspect::{EguiInspect, InspectNumber};
use epaint::{Color32, ColorImage, TextureHandle};
use ndarray::{Array2, Array3};
use ocl::{OclPrm, Platform, ProQue};
use simple_ocl::{prog_que_from_source_path, PairedBuffers2, PairedBuffers3};
use std::{
    error::Error,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

#[repr(C)]
#[derive(EguiInspect, Clone, Copy, Debug, Default, PartialEq)]
struct BBox {
    #[inspect(min=-2.0, max=2.0)]
    left: f64,
    #[inspect(min=-2.0, max=2.0)]
    right: f64,
    #[inspect(min=-2.0, max=2.0)]
    bot: f64,
    #[inspect(min=-2.0, max=2.0)]
    top: f64,
}

unsafe impl OclPrm for BBox {}

#[repr(C)]
#[derive(Debug, EguiInspect, PartialEq, Clone, Copy)]
struct Complex {
    #[inspect(min=-2.0, max=2.0)]
    re: f64,
    #[inspect(min=-2.0, max=2.0)]
    im: f64,
}

impl Default for Complex {
    fn default() -> Self {
        Self { re: -0.7, im: 0.3 }
    }
}

#[repr(C)]
#[derive(Debug, EguiInspect, PartialEq, Clone, Copy)]
struct Freqs {
    #[inspect(log_slider, min = 1.0, max = 1000.0)]
    f1: f64,
    #[inspect(log_slider, min = 1.0, max = 1000.0)]
    f2: f64,
    #[inspect(log_slider, min = 1.0, max = 1000.0)]
    f3: f64,
}

impl Default for Freqs {
    fn default() -> Self {
        Self {
            f1: 1.1,
            f2: 3.3,
            f3: 9.9,
        }
    }
}

unsafe impl OclPrm for Freqs {}

#[derive(Default, EguiInspect, PartialEq, Clone)]
enum FractalMode {
    #[default]
    Mandel,
    Julia {
        c: Complex,
    },
}

impl FractalMode {
    fn get_c(&self) -> Complex {
        match self {
            FractalMode::Mandel => Complex::default(),
            FractalMode::Julia { c } => c.clone(),
        }
    }
}

/// Shared fractal params
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct SFParam {
    mode_int: i32,
    c: Complex,
    view: BBox,
    max_iter: i32,
}

unsafe impl OclPrm for SFParam {}

/// UI for Shared fractal params
#[derive(EguiInspect, Clone, PartialEq)]
struct SFParamUI {
    mode: FractalMode,
    center: Complex,
    #[inspect(log_slider, min = 1e-19, max = 1.0)]
    zoom: f64,
    #[inspect(log_slider, min = 0.1, max = 10.0)]
    aspect: f64,
    #[inspect(log_slider, min = 1.0, max = 10000.0)]
    max_iter: i32,
}

impl Default for SFParamUI {
    fn default() -> Self {
        Self {
            max_iter: 100,
            center: Complex { re: -0.4, im: 0.0 },
            mode: Default::default(),
            zoom: 1.0,
            aspect: 1.0,
        }
    }
}

impl SFParamUI {
    fn get_view_bbox(&self) -> BBox {
        let delta_re = self.zoom;
        let delta_im = self.zoom * self.aspect;
        BBox {
            left: self.center.re - delta_re,
            right: self.center.re + delta_re,
            bot: self.center.im - delta_im,
            top: self.center.im + delta_im,
        }
    }

    fn get_c_struct(&self) -> SFParam {
        SFParam {
            mode_int: match self.mode {
                FractalMode::Mandel => 1,
                FractalMode::Julia { .. } => 0,
            },
            c: self.mode.get_c(),
            view: self.get_view_bbox(),
            max_iter: self.max_iter,
        }
    }
}

struct ItersImage {
    mat_dims: (usize, usize),
    cimage: ColorImage,
    texture: Option<TextureHandle>,
}

impl ItersImage {
    fn new(mat_dims: (usize, usize)) -> Self {
        Self {
            mat_dims,
            texture: None,
            cimage: Default::default(),
        }
    }

    fn update(&mut self, rgb: &Array3<u8>) {
        self.cimage =
            ColorImage::from_rgb([self.mat_dims.1, self.mat_dims.0], rgb.as_slice().unwrap());
        self.texture = None;
    }
}

impl EguiInspect for ItersImage {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        let handle: &egui::TextureHandle = self.texture.get_or_insert_with(|| {
            ui.ctx()
                .load_texture("test_img", self.cimage.clone(), Default::default())
        });
        ui.image(handle);
    }
}

struct OCLHelper {
    pro_que: ProQue,
    field_1: PairedBuffers2<f64>,
    rgb: PairedBuffers3<u8>,
}

impl OCLHelper {
    fn new(im_mat_dims: (usize, usize)) -> Self {
        let mut pro_que =
            prog_que_from_source_path("./src/ocl/mandel.cl", vec!["-I./src/ocl".to_string()]);
        let field_1 = PairedBuffers2::create_from(Array2::<f64>::zeros(im_mat_dims), &mut pro_que);
        let (n, m) = im_mat_dims;
        let rgb = PairedBuffers3::create_from(Array3::<u8>::zeros((n, m, 3)), &mut pro_que);
        OCLHelper {
            pro_que,
            field_1,
            rgb,
        }
    }

    fn arc_mut_new(im_mat_dims: (usize, usize)) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::new(im_mat_dims)))
    }

    fn run_escape_iter(&mut self, fparam: SFParam) -> ocl::Result<()> {
        let kernel = self
            .pro_que
            .kernel_builder("escape_iter_fpn")
            .arg(&self.field_1.device)
            .arg(fparam)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }

    fn run_min_prox(&mut self, fparam: SFParam, prox_type: ProxType) -> ocl::Result<()> {
        let kernel = self
            .pro_que
            .kernel_builder("min_prox")
            .arg(&self.field_1.device)
            .arg(fparam)
            .arg(prox_type)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }

    fn run_box_trap_partial(&mut self, fparam: SFParam, box_: BBox, real: bool) -> ocl::Result<()> {
        let kernel_name = if real {
            "orbit_trap_re"
        } else {
            "orbit_trap_im"
        };

        let kernel = self
            .pro_que
            .kernel_builder(kernel_name)
            .arg(&self.field_1.device)
            .arg(fparam)
            .arg(box_)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }

    fn run_map_sines(&mut self, freqs: Freqs) -> ocl::Result<()> {
        let kernel = self
            .pro_que
            .kernel_builder("map_sines")
            .arg(&self.field_1.device)
            .arg(&self.rgb.device)
            .arg(freqs)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Default, EguiInspect, PartialEq, Clone, Copy)]
struct ProxType {
    to_unit_circ: bool,
    to_horizontal: bool,
    to_vertical: bool,
}

unsafe impl OclPrm for ProxType {}

/// Scalar field selection for visualisation channels
#[derive(Clone, PartialEq, EguiInspect, Default)]
enum FractalFieldType {
    #[default]
    ItersToEscape,
    Proximity {
        prox_type: ProxType,
    },
    BoxTrapRe {
        #[inspect(name = "box")]
        box_: BBox,
    },
    BoxTrapIm {
        #[inspect(name = "box")]
        box_: BBox,
    },
}

#[derive(Clone, PartialEq, EguiInspect)]
enum FractalVisualisationType {
    SingleFieldCmaped {
        field_type: FractalFieldType,
        freqs: Freqs,
    },
    DualFieldImageMap,
    TriFieldRGB,
}

impl Default for FractalVisualisationType {
    fn default() -> Self {
        Self::SingleFieldCmaped {
            field_type: Default::default(),
            freqs: Default::default(),
        }
    }
}

#[derive(Clone, Default, PartialEq, EguiInspect)]
struct FractalParams {
    sfparam: SFParamUI,
    fields: FractalVisualisationType,
}

type ThreadResult = Result<(), String>;

#[derive(EguiInspect)]
struct FractalViewer {
    fp: FractalParams,
    #[inspect(hide)]
    old_fp: FractalParams,
    iters_image: ItersImage,
    #[inspect(hide)]
    ocl_helper: Arc<Mutex<OCLHelper>>,
    #[inspect(hide)]
    join_handle: Option<JoinHandle<ThreadResult>>,
}

impl FractalViewer {
    fn new() -> Self {
        let im_mat_dims = (768, 1280);
        let mut old_fp = FractalParams::default();
        old_fp.sfparam.max_iter = 0;

        Self {
            iters_image: ItersImage::new(im_mat_dims),
            ocl_helper: OCLHelper::arc_mut_new(im_mat_dims),
            join_handle: None,
            fp: Default::default(),
            old_fp,
        }
    }

    fn run_kernel_in_background(&mut self) {
        let helper_arc = self.ocl_helper.clone();
        let frac_param = self.fp.clone();
        let dims = self.iters_image.mat_dims;

        self.join_handle = Some(std::thread::spawn(move || {
            let FractalParams { sfparam, fields } = frac_param;
            let sfparam_c = sfparam.get_c_struct();
            match helper_arc.try_lock() {
                Ok(mut guard) => match fields {
                    FractalVisualisationType::SingleFieldCmaped { field_type, freqs } => {
                        guard.pro_que.set_dims(dims);
                        match field_type {
                            FractalFieldType::ItersToEscape => {
                                guard.run_escape_iter(sfparam_c)?;
                            }
                            FractalFieldType::Proximity { prox_type } => {
                                guard.run_min_prox(sfparam_c, prox_type)?;
                            }
                            FractalFieldType::BoxTrapRe { box_ } => {
                                guard.run_box_trap_partial(sfparam_c, box_, true)?;
                            }
                            FractalFieldType::BoxTrapIm { box_ } => {
                                guard.run_box_trap_partial(sfparam_c, box_, false)?;
                            }
                        }
                        guard.run_map_sines(freqs)?;
                        guard.rgb.from_device()?;
                        Ok(())
                    }
                    FractalVisualisationType::DualFieldImageMap => todo!(),
                    FractalVisualisationType::TriFieldRGB => todo!(),
                },
                Err(_) => Err("mutex is locked".to_string()),
            }
        }));
    }

    fn collect_result(&mut self) {
        let handle = self.join_handle.take().unwrap();
        match handle.join().expect("thread join error") {
            Ok(_) => match self.ocl_helper.try_lock() {
                Ok(guard) => self.iters_image.update(&guard.rgb.host),
                Err(err) => println!("could not aquire mutex in update: {err}"),
            },
            Err(err) => println!("Error on other thread: {}", err),
        }
    }
}

impl eframe::App for FractalViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let job_still_running = match &mut self.join_handle {
            Some(handle) => {
                if handle.is_finished() {
                    self.collect_result();
                    false
                } else {
                    true
                }
            }
            None => false,
        };

        let mut status_text = RichText::new("GPU Busy").color(Color32::RED);
        let params_updated = self.old_fp != self.fp;
        if !job_still_running {
            if params_updated {
                self.run_kernel_in_background();
                self.old_fp = self.fp.clone();
            } else {
                status_text = RichText::new("GPU Waiting").color(Color32::GREEN);
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(status_text);

            self.inspect_mut("", ui)
        });
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Note: Work around to strange segfault issue when building proque in eframe::App
    dbg!(Platform::default());

    eframe::run_native(
        "Fractal viewer",
        NativeOptions::default(),
        Box::new(|_cc| Box::new(FractalViewer::new())),
    )?;
    Ok(())
}
