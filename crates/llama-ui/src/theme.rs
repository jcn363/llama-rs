// Custom button style for model picker
use iced::{widget::button, widget::button::StyleSheet, Theme, Color};

#[derive(Debug, Clone, Copy)]
pub struct ModelButton;

impl button::StyleSheet for ModelButton {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> button::Appearance {
        button::Appearance {
            background: Some(Color::from_rgba8(0x4A, 0x90, 0xE2, 0.9).into()),
            text_color: Color::WHITE,
            border_radius: 6.0,
            border_width: 1.0,
            border_color: Color::from_rgba8(0x30, 0x60, 0xA0, 1.0),
            ..Default::default()
        }
    }
    fn hovered(&self, _style: &Self::Style) -> button::Appearance {
        let mut a = self.active(_style);
        a.background = Some(Color::from_rgba8(0x5A, 0xA0, 0xF2, 1.0).into());
        a
    }
    fn pressed(&self, _style: &Self::Style) -> button::Appearance {
        let mut a = self.active(_style);
        a.background = Some(Color::from_rgba8(0x3A, 0x80, 0xC2, 1.0).into());
        a
    }
    fn disabled(&self, _style: &Self::Style) -> button::Appearance {
        let mut a = self.active(_style);
        a.background = Some(Color::from_rgba8(0x80, 0x80, 0x80, 0.5).into());
        a.text_color = Color::from_rgba8(0xC0, 0xC0, 0xC0, 0.7);
        a
    }
}
