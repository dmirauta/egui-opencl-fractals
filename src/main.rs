extern crate ocl;
use eframe::NativeOptions;
use egui::{RichText, Window};
use egui_extras::syntax_highlighting::{highlight, CodeTheme};
use egui_inspect::{EguiInspect, InspectNumber};
use epaint::{Color32, ColorImage, TextureHandle};
use image::{io::Reader, EncodableLayout, ImageResult};
use ndarray::{Array2, Array3};
use ocl::{Platform, ProQue};
use simple_ocl::{try_prog_que_from_source, PairedBuffers2, PairedBuffers3};
use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

mod wrapper_types;
use wrapper_types::{BBox, Complex, Freqs, ImDims, ProxType, SFParam};

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

/// UI for Shared fractal params
#[derive(EguiInspect, Clone, PartialEq)]
struct SFParamUI {
    mode: FractalMode,
    view_center: Complex,
    #[inspect(log_slider, min = 1e-19, max = 2.0)]
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
            view_center: Complex { re: -0.4, im: 0.0 },
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
            left: self.view_center.re - delta_re,
            right: self.view_center.re + delta_re,
            bot: self.view_center.im - delta_im,
            top: self.view_center.im + delta_im,
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
                .load_texture("fractal", self.cimage.clone(), Default::default())
        });
        ui.image(handle);
    }
}

// ocl source baked into binary at build time
static OCL_STRUCTS: &str = include_str!("./ocl/mandelstructs.h");
static OCL_FUNCS: &str = include_str!("./ocl/mandelutils.c");
static OCL_KERNELS: &str = include_str!("./ocl/mandel.cl");

fn insert_custom_func(custom_func: String) -> String {
    let funcs = OCL_FUNCS.to_string();
    let mut sp = funcs.split("//>>");
    let [before_func, remainder] = [sp.next().unwrap(), sp.next().unwrap()];
    let rs = remainder.to_string();
    let mut sp = rs.split("//<<");
    let [_, after_func] = [sp.next().unwrap(), sp.next().unwrap()];
    format!("{before_func}{custom_func}{after_func}")
}

struct OCLHelper {
    pro_que: ProQue,
    field_1: PairedBuffers2<f64>,
    field_2: PairedBuffers2<f64>,
    field_3: PairedBuffers2<f64>,
    sampled_path: Option<PathBuf>,
    sampled_rgb: Option<PairedBuffers3<u8>>,
    rgb: PairedBuffers3<u8>,
}

impl OCLHelper {
    fn new(im_mat_dims: (usize, usize), custom_iter_func: Option<String>) -> ocl::Result<Self> {
        let ocl_funcs_custom = match custom_iter_func {
            Some(cf) => insert_custom_func(cf),
            None => OCL_FUNCS.to_string(),
        };
        let full_source = format!("{OCL_STRUCTS}{ocl_funcs_custom}{OCL_KERNELS}");
        let mut pro_que =
            try_prog_que_from_source(full_source, "mandel", vec!["-DEXTERNAL_CONCAT".to_string()])?;
        let field_1 = PairedBuffers2::create_from(Array2::<f64>::zeros(im_mat_dims), &mut pro_que);
        let field_2 = PairedBuffers2::create_from(Array2::<f64>::zeros(im_mat_dims), &mut pro_que);
        let field_3 = PairedBuffers2::create_from(Array2::<f64>::zeros(im_mat_dims), &mut pro_que);
        let (n, m) = im_mat_dims;
        let rgb = PairedBuffers3::create_from(Array3::<u8>::zeros((n, m, 3)), &mut pro_que);
        Ok(OCLHelper {
            pro_que,
            field_1,
            field_2,
            field_3,
            rgb,
            sampled_path: None,
            sampled_rgb: None,
        })
    }

    fn update_sampled(&mut self, ip: PathBuf) {
        match load_decoded(&ip) {
            Ok(sampled) => {
                let pb = PairedBuffers3::create_from(sampled, &mut self.pro_que);
                pb.to_device().unwrap();
                self.sampled_rgb = Some(pb);
                self.sampled_path = Some(ip);
            }
            Err(err) => println!("{err}"),
        }
    }

    fn field_ref(&self, i: usize) -> &ocl::Buffer<f64> {
        match i {
            1 => &self.field_1.device,
            2 => &self.field_2.device,
            3 => &self.field_3.device,
            _ => {
                panic!("invalid field index");
            }
        }
    }

    fn run_escape_iter(&mut self, fi: usize, fparam: SFParam) -> ocl::Result<()> {
        let kernel = self
            .pro_que
            .kernel_builder("escape_iter_fpn")
            .arg(self.field_ref(fi))
            .arg(fparam)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }

    fn run_min_prox(&mut self, fi: usize, fparam: SFParam, prox_type: ProxType) -> ocl::Result<()> {
        let kernel = self
            .pro_que
            .kernel_builder("min_prox")
            .arg(self.field_ref(fi))
            .arg(fparam)
            .arg(prox_type)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }

    fn run_box_trap_partial(
        &mut self,
        fi: usize,
        fparam: SFParam,
        box_: BBox,
        real: bool,
    ) -> ocl::Result<()> {
        let kernel_name = if real {
            "orbit_trap_re"
        } else {
            "orbit_trap_im"
        };

        let kernel = self
            .pro_que
            .kernel_builder(kernel_name)
            .arg(self.field_ref(fi))
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

    fn run_pack(&mut self, normalise: bool) -> ocl::Result<()> {
        let kernel_name = if normalise { "pack_norm" } else { "pack" };
        let kernel = self
            .pro_que
            .kernel_builder(kernel_name)
            .arg(&self.field_1.device)
            .arg(&self.field_2.device)
            .arg(&self.field_3.device)
            .arg(&self.rgb.device)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        Ok(())
    }

    fn run_map_img(&mut self) -> ocl::Result<()> {
        if let Some(sampled) = self.sampled_rgb.as_ref() {
            let s = sampled.host.shape();
            let imdims = ImDims {
                height: s[0] as i32,
                width: s[1] as i32,
            };
            let kernel = self
                .pro_que
                .kernel_builder("map_img2")
                .arg(&self.field_1.device)
                .arg(&self.field_2.device)
                .arg(&sampled.device)
                .arg(&self.rgb.device)
                .arg(imdims)
                .build()?;

            unsafe {
                kernel.enq()?;
            }
        }

        Ok(())
    }
}

/// Scalar field selection for visualisation channels
#[derive(Clone, PartialEq, EguiInspect, Default)]
enum FractalFieldType {
    #[default]
    ItersToEscape,
    ChainMinProximity {
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

fn load_decoded(fpath: impl AsRef<Path>) -> ImageResult<Array3<u8>> {
    let img = Reader::open(fpath)?
        .with_guessed_format()?
        .decode()?
        .into_rgb8();
    let (w, h) = img.dimensions();
    let h = h as usize;
    let w = w as usize;
    let mut sampled = Array3::<u8>::zeros((h, w, 3));
    sampled
        .as_slice_mut()
        .unwrap()
        .copy_from_slice(img.as_bytes());
    Ok(sampled)
}

#[derive(Clone, Default)]
struct SelectedImage {
    path: Option<PathBuf>,
    texture: Option<TextureHandle>,
}

impl PartialEq for SelectedImage {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl EguiInspect for SelectedImage {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        if ui.button("load sampled image").clicked() {
            if let Some(fpath) = rfd::FileDialog::new().set_directory(".").pick_file() {
                let _ = self.path.insert(fpath.clone());
                match load_decoded(fpath) {
                    Ok(img) => {
                        let s = img.shape();
                        let cimage = ColorImage::from_rgb([s[1], s[0]], img.as_slice().unwrap());
                        let _ = self.texture.insert(ui.ctx().load_texture(
                            "fractal",
                            cimage,
                            Default::default(),
                        ));
                    }
                    Err(err) => println!("{err}"),
                }
            }
        }

        if let Some(tex) = &self.texture {
            Window::new("sampled image").show(ui.ctx(), |ui| {
                ui.image(tex);
            });
        }
    }
}

#[derive(Clone, PartialEq, EguiInspect)]
enum FractalVisualisationType {
    SingleFieldCmaped {
        field_type: FractalFieldType,
        cmap_freqs: Freqs,
    },
    DualFieldImageMap {
        u_field_type: FractalFieldType,
        v_field_type: FractalFieldType,
        selected_image: SelectedImage,
    },
    TriFieldRGB {
        r_field_type: FractalFieldType,
        g_field_type: FractalFieldType,
        b_field_type: FractalFieldType,
        normalise_colors: bool,
    },
}

impl Default for FractalVisualisationType {
    fn default() -> Self {
        Self::SingleFieldCmaped {
            field_type: Default::default(),
            cmap_freqs: Default::default(),
        }
    }
}

#[derive(Clone, Default, PartialEq, EguiInspect)]
struct FractalParams {
    #[inspect(name = "Shared")]
    sfparam: SFParamUI,
    vis_type: FractalVisualisationType,
}

/// Basic editing with syntax highlighting through egui_extras, better highlighting available
/// in syntect
struct FunctionEditor {
    code: String,
    theme: CodeTheme,
}

impl Default for FunctionEditor {
    fn default() -> Self {
        Self {
            code: "// Define custom iteration function f: (Complex_t, Complex_t) -> Complex_t
// first argument is spatially dependent while the second is either 
// z_0 for mandel-like and user input for julia-like options.
// Complex_t has fields re and im. Convenience functions
// `complex_add: (Complex_t, Complex_t) -> Complex_t`
// `complex_mult: (Complex_t, Complex_t) -> Complex_t`
// and `complex_pow: (Complex_t, int) -> Complex_t` are in scope.
inline Complex_t f(Complex_t z, Complex_t c) {
  return complex_add(complex_pow(z, 2), c);
}"
            .to_string(),
            theme: Default::default(),
        }
    }
}

impl EguiInspect for FunctionEditor {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        let mut layouter = |ui: &egui::Ui, string: &str, wrap_width: f32| {
            let mut layout_job = highlight(ui.ctx(), &self.theme, string, "c");
            layout_job.wrap.max_width = wrap_width;
            ui.fonts(|f| f.layout_job(layout_job))
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut self.code)
                    .font(egui::TextStyle::Monospace) // for cursor height
                    .code_editor()
                    .desired_rows(10)
                    .lock_focus(true)
                    .desired_width(f32::INFINITY)
                    .layouter(&mut layouter),
            );
        });
    }
}

type ThreadResult = Result<(), String>;

struct FractalViewer {
    fp: FractalParams,
    old_fp: FractalParams,
    editor: FunctionEditor,
    error: Option<String>,
    iters_image: ItersImage,
    ocl_helper: Arc<Mutex<OCLHelper>>,
    join_handle: Option<JoinHandle<ThreadResult>>,
}

impl FractalViewer {
    fn new() -> Self {
        let im_mat_dims = (768, 1280);
        let mut old_fp = FractalParams::default();
        old_fp.sfparam.max_iter = 0;

        Self {
            editor: Default::default(),
            iters_image: ItersImage::new(im_mat_dims),
            ocl_helper: Arc::new(Mutex::new(OCLHelper::new(im_mat_dims, None).unwrap())),
            join_handle: None,
            fp: Default::default(),
            old_fp,
            error: None,
        }
    }

    fn handle_field(
        fi: usize,
        helper: &mut OCLHelper,
        field_type: FractalFieldType,
        sfparam_c: SFParam,
    ) -> ocl::Result<()> {
        match field_type {
            FractalFieldType::ItersToEscape => {
                helper.run_escape_iter(fi, sfparam_c)?;
            }
            FractalFieldType::ChainMinProximity { prox_type } => {
                helper.run_min_prox(fi, sfparam_c, prox_type)?;
            }
            FractalFieldType::BoxTrapRe { box_ } => {
                helper.run_box_trap_partial(fi, sfparam_c, box_, true)?;
            }
            FractalFieldType::BoxTrapIm { box_ } => {
                helper.run_box_trap_partial(fi, sfparam_c, box_, false)?;
            }
        }
        Ok(())
    }

    fn run_kernel_in_background(&mut self) {
        let helper_arc = self.ocl_helper.clone();
        let frac_param = self.fp.clone();
        let dims = self.iters_image.mat_dims;

        self.join_handle = Some(std::thread::spawn(move || {
            let FractalParams { sfparam, vis_type } = frac_param;
            let sfparam_c = sfparam.get_c_struct();
            match helper_arc.try_lock() {
                Ok(mut guard) => {
                    guard.pro_que.set_dims(dims);
                    match vis_type {
                        FractalVisualisationType::SingleFieldCmaped {
                            field_type,
                            cmap_freqs: freqs,
                        } => {
                            Self::handle_field(1, &mut guard, field_type, sfparam_c)?;
                            guard.run_map_sines(freqs)?;
                        }
                        FractalVisualisationType::DualFieldImageMap {
                            u_field_type,
                            v_field_type,
                            selected_image,
                        } => {
                            if guard.sampled_path != selected_image.path {
                                if let Some(ip) = &selected_image.path {
                                    guard.update_sampled(ip.clone());
                                    // update_sampled changes que size
                                    guard.pro_que.set_dims(dims);
                                }
                            };
                            if guard.sampled_rgb.is_some() {
                                Self::handle_field(1, &mut guard, u_field_type, sfparam_c)?;
                                Self::handle_field(2, &mut guard, v_field_type, sfparam_c)?;
                                // TODO: not all field types normalised for UV coords
                                guard.run_map_img()?;
                            }
                        }
                        FractalVisualisationType::TriFieldRGB {
                            r_field_type,
                            g_field_type,
                            b_field_type,
                            normalise_colors,
                        } => {
                            Self::handle_field(1, &mut guard, r_field_type, sfparam_c)?;
                            Self::handle_field(2, &mut guard, g_field_type, sfparam_c)?;
                            Self::handle_field(3, &mut guard, b_field_type, sfparam_c)?;
                            guard.run_pack(normalise_colors)?;
                        }
                    };
                    guard.rgb.from_device()?;
                    Ok(())
                }
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

    fn try_recompile(&mut self) {
        if self.join_handle.is_none() {
            if let Ok(mut guard) = self.ocl_helper.try_lock() {
                match OCLHelper::new(self.iters_image.mat_dims, Some(self.editor.code.clone())) {
                    Ok(new_helper) => {
                        *guard = new_helper;
                        self.old_fp.sfparam.max_iter = 0; // trigger recompute
                        self.error = None;
                    }
                    Err(err) => self.error = Some(format!("{err}")),
                }
            }
        } else {
            self.error = Some(format!("Could not recompile, GPU was busy."));
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
            Window::new("Controls").show(ctx, |ui| {
                self.fp.inspect_mut("Fractal parameters", ui);
            });
            Window::new("Custom function").show(ctx, |ui| {
                self.editor.inspect_mut("Custom function", ui);
                if ui.button("Recompile").clicked() {
                    self.try_recompile();
                }
                if let Some(err) = &self.error {
                    ui.label(err.as_str());
                }
            });
            self.iters_image.inspect_mut("", ui);
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
