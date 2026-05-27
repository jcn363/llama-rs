// Custom button style for model picker
// Uses iced 0.13's `button::StyleFn` (closure-based) styling API.
use iced::widget::button::{self, Status};
use iced::{Color, Theme};

/// Returns a button style function for a primary-action button (blue).
///
/// Usage:
/// ```ignore
/// button("Click me").style(primary_button_style)
/// ```
#[must_use]
pub fn primary_button_style(_theme: &Theme, status: Status) -> button::Style {
    let base = button::Style {
        background: Some(Color::from_rgba8(0x4A, 0x90, 0xE2, 0.9).into()),
        text_color: Color::WHITE,
        border: iced::Border {
            radius: 6.0.into(),
            width: 1.0,
            color: Color::from_rgba8(0x30, 0x60, 0xA0, 1.0),
        },
        shadow: iced::Shadow::default(),
    };

    match status {
        Status::Active | Status::Pressed => base,
        Status::Hovered => button::Style {
            background: Some(Color::from_rgba8(0x5A, 0xA0, 0xF2, 1.0).into()),
            ..base
        },
        Status::Disabled => button::Style {
            background: Some(Color::from_rgba8(0x80, 0x80, 0x80, 0.5).into()),
            text_color: Color::from_rgba8(0xC0, 0xC0, 0xC0, 0.7),
            ..base
        },
    }
}

/// Helper to create a boxed style function for use with `iced::widget::Button`.
#[must_use]
pub fn primary_button_style_fn() -> iced::widget::button::StyleFn<'static, Theme> {
    Box::new(primary_button_style)
}
