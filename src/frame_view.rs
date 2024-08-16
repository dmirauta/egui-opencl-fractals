// TODO: Should be editing the image data of the texturehandle on the gpu?

use egui_inspect::egui::{self, ColorImage, Image, TextureHandle};
use egui_inspect::EguiInspect;
use ndarray::Array3;

pub struct FrameView {
    pub dims: (usize, usize),
    texture: Option<TextureHandle>,
}

impl FrameView {
    pub fn new(dims: (usize, usize)) -> Self {
        Self {
            dims,
            texture: None,
        }
    }

    pub fn update(&mut self, rgb: &Array3<u8>) {
        if let Some(handle) = &mut self.texture {
            let cimage = ColorImage::from_rgb([self.dims.1, self.dims.0], rgb.as_slice().unwrap());
            handle.set(cimage, Default::default());
        }
    }
}

impl EguiInspect for FrameView {
    fn inspect(&self, _label: &str, _ui: &mut egui::Ui) {
        todo!()
    }

    fn inspect_mut(&mut self, _label: &str, ui: &mut egui::Ui) {
        let handle: &egui::TextureHandle = self.texture.get_or_insert_with(|| {
            let cimage = ColorImage::from_rgb(
                [self.dims.1, self.dims.0],
                vec![0; self.dims.0 * self.dims.1 * 3].as_slice(),
            );
            ui.ctx()
                .load_texture("fractal", cimage.clone(), Default::default())
        });
        ui.add(Image::new(handle).shrink_to_fit());
    }
}
