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

/// Returns a button style function for a secondary-action button (gray).
#[must_use]
pub fn secondary_button_style(_theme: &Theme, status: Status) -> button::Style {
    let base = button::Style {
        background: Some(Color::from_rgba8(0x60, 0x60, 0x60, 0.8).into()),
        text_color: Color::WHITE,
        border: iced::Border {
            radius: 6.0.into(),
            width: 1.0,
            color: Color::from_rgba8(0x40, 0x40, 0x40, 1.0),
        },
        shadow: iced::Shadow::default(),
    };

    match status {
        Status::Active | Status::Pressed => base,
        Status::Hovered => button::Style {
            background: Some(Color::from_rgba8(0x70, 0x70, 0x70, 1.0).into()),
            ..base
        },
        Status::Disabled => button::Style {
            background: Some(Color::from_rgba8(0x80, 0x80, 0x80, 0.5).into()),
            text_color: Color::from_rgba8(0xC0, 0xC0, 0xC0, 0.7),
            ..base
        },
    }
}

/// Returns a button style function for a danger/action button (red).
#[must_use]
pub fn danger_button_style(_theme: &Theme, status: Status) -> button::Style {
    let base = button::Style {
        background: Some(Color::from_rgba8(0xE2, 0x4A, 0x4A, 0.9).into()),
        text_color: Color::WHITE,
        border: iced::Border {
            radius: 6.0.into(),
            width: 1.0,
            color: Color::from_rgba8(0xA0, 0x30, 0x30, 1.0),
        },
        shadow: iced::Shadow::default(),
    };

    match status {
        Status::Active | Status::Pressed => base,
        Status::Hovered => button::Style {
            background: Some(Color::from_rgba8(0xF2, 0x5A, 0x5A, 1.0).into()),
            ..base
        },
        Status::Disabled => button::Style {
            background: Some(Color::from_rgba8(0x80, 0x80, 0x80, 0.5).into()),
            text_color: Color::from_rgba8(0xC0, 0xC0, 0xC0, 0.7),
            ..base
        },
    }
}

/// Returns a button style function for a success button (green).
#[must_use]
pub fn success_button_style(_theme: &Theme, status: Status) -> button::Style {
    let base = button::Style {
        background: Some(Color::from_rgba8(0x4A, 0xE2, 0x6A, 0.9).into()),
        text_color: Color::WHITE,
        border: iced::Border {
            radius: 6.0.into(),
            width: 1.0,
            color: Color::from_rgba8(0x30, 0xA0, 0x40, 1.0),
        },
        shadow: iced::Shadow::default(),
    };

    match status {
        Status::Active | Status::Pressed => base,
        Status::Hovered => button::Style {
            background: Some(Color::from_rgba8(0x5A, 0xF2, 0x7A, 1.0).into()),
            ..base
        },
        Status::Disabled => button::Style {
            background: Some(Color::from_rgba8(0x80, 0x80, 0x80, 0.5).into()),
            text_color: Color::from_rgba8(0xC0, 0xC0, 0xC0, 0.7),
            ..base
        },
    }
}

/// Returns a container style for chat message bubbles (user messages - blue tint).
#[must_use]
pub fn user_message_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Color::from_rgba8(0x4A, 0x90, 0xE2, 0.15).into()),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: Color::from_rgba8(0x4A, 0x90, 0xE2, 0.3),
        },
        text_color: None,
        shadow: iced::Shadow::default(),
    }
}

/// Returns a container style for chat message bubbles (assistant messages - green tint).
#[must_use]
pub fn assistant_message_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Color::from_rgba8(0x4A, 0xE2, 0x6A, 0.15).into()),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: Color::from_rgba8(0x4A, 0xE2, 0x6A, 0.3),
        },
        text_color: None,
        shadow: iced::Shadow::default(),
    }
}

/// Returns a container style for system messages (gray tint).
#[must_use]
pub fn system_message_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Color::from_rgba8(0x80, 0x80, 0x80, 0.15).into()),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: Color::from_rgba8(0x80, 0x80, 0x80, 0.3),
        },
        text_color: None,
        shadow: iced::Shadow::default(),
    }
}

/// Returns a container style for status bar (dark background).
#[must_use]
pub fn status_bar_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Color::from_rgba8(0x2A, 0x2A, 0x2A, 0.9).into()),
        border: iced::Border {
            radius: 0.0.into(),
            width: 0.0,
            color: Color::TRANSPARENT,
        },
        text_color: Some(Color::from_rgba8(0xC0, 0xC0, 0xC0, 1.0)),
        shadow: iced::Shadow::default(),
    }
}

/// Returns a container style for the main content area.
#[must_use]
pub fn content_area_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Color::from_rgba8(0x1A, 0x1A, 0x1A, 1.0).into()),
        border: iced::Border {
            radius: 0.0.into(),
            width: 0.0,
            color: Color::TRANSPARENT,
        },
        text_color: Some(Color::from_rgba8(0xE0, 0xE0, 0xE0, 1.0)),
        shadow: iced::Shadow::default(),
    }
}

/// Helper to create a boxed style function for use with `iced::widget::Button`.
#[must_use]
pub fn primary_button_style_fn() -> iced::widget::button::StyleFn<'static, Theme> {
    Box::new(primary_button_style)
}

/// Helper to create a boxed secondary style function.
#[must_use]
pub fn secondary_button_style_fn() -> iced::widget::button::StyleFn<'static, Theme> {
    Box::new(secondary_button_style)
}

/// Helper to create a boxed danger style function.
#[must_use]
pub fn danger_button_style_fn() -> iced::widget::button::StyleFn<'static, Theme> {
    Box::new(danger_button_style)
}

/// Helper to create a boxed success style function.
#[must_use]
pub fn success_button_style_fn() -> iced::widget::button::StyleFn<'static, Theme> {
    Box::new(success_button_style)
}
