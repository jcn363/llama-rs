---
name: web-accessibility
description: "Use when designing accessible UI components, auditing existing sites for accessibility issues, implementing screen reader-friendly forms, managing focus in modals/dropdowns, or meeting WCAG compliance requirements - covers semantic HTML, keyboard navigation, ARIA attributes, color contrast, and testing with axe-core and Lighthouse for React/TypeScript components."
---

# Web Accessibility (A11y)

## Overview

Implement WCAG 2.1 accessibility standards for semantic HTML, keyboard navigation, ARIA attributes, color contrast, and screen reader support — ensuring all users can interact with your application regardless of ability.

## When to Use

- **New UI Component Development**: Designing accessible components from the start
- **Accessibility Audit**: Identifying and fixing accessibility issues in existing sites
- **Form Implementation**: Writing screen reader-friendly forms
- **Modals/Dropdowns**: Focus management and keyboard trap prevention
- **WCAG Compliance**: Meeting legal requirements or standards

## Input Requirements

### Required
- **Framework**: React, Vue, Svelte, Vanilla JS, etc.
- **Component Type**: Button, Form, Modal, Dropdown, Navigation, etc.
- **WCAG Level**: A, AA, AAA (default: AA)

### Optional
- Screen Reader: NVDA, JAWS, VoiceOver (for testing)
- Automated Testing Tool: axe-core, Pa11y, Lighthouse (default: axe-core)
- Browser: Chrome, Firefox, Safari (default: Chrome)

### Input Example

> Make a React modal component accessible:
> - Framework: React + TypeScript
> - WCAG Level: AA
> - Requirements:
>   - Focus trap (focus stays inside the modal)
>   - Close with ESC key
>   - Close by clicking the background
>   - Title/description read by screen readers

## Core Workflow

### Step 1: Use Semantic HTML

Use meaningful HTML elements to make the structure and purpose clear to assistive technologies.

**Tasks:**
- Use semantic tags: `<button>`, `<nav>`, `<main>`, `<header>`, `<footer>`, etc.
- Avoid overusing `<div>` and `<span>` for interactive elements
- Use heading hierarchy (`<h1>` → `<h6>`) correctly — never skip levels
- Connect `<label>` with `<input>` using `for`/`id` or wrapping

**Example (❌ Bad vs ✅ Good):**

```html
<!-- ❌ Bad: div/span soup -->
<div class="header">
  <span class="title">My App</span>
  <div class="nav">
    <div class="nav-item" onclick="navigate()">Home</div>
  </div>
</div>

<!-- ✅ Good: semantic HTML -->
<header>
  <h1>My App</h1>
  <nav aria-label="Main navigation">
    <ul>
      <li><a href="/">Home</a></li>
      <li><a href="/about">About</a></li>
    </ul>
  </nav>
</header>
```

**Form labels:**

```html
<!-- ❌ Bad: no label -->
<input type="text" placeholder="Enter your name">

<!-- ✅ Good: label connected via for/id -->
<label for="name">Name:</label>
<input type="text" id="name" name="name" required>

<!-- ✅ Good: label wrapping input -->
<label>
  Email:
  <input type="email" name="email" required>
</label>
```

### Step 2: Implement Keyboard Navigation

Ensure all features are usable without a mouse.

**Tasks:**
- Move focus with Tab and Shift+Tab
- Activate buttons with Enter/Space
- Navigate lists/menus with arrow keys
- Close modals/dropdowns with ESC
- Use `tabindex` appropriately

**Decision criteria for `tabindex`:**
| Value | Behavior |
|-------|----------|
| `0` | Focusable via Tab, follows DOM order |
| `-1` | Not reachable via Tab, but programmatically focusable |
| `>0` | Avoid — changes natural focus order |

**Example (React Dropdown with keyboard nav):**

```tsx
import React, { useState, useRef, useEffect } from 'react';

function AccessibleDropdown({ label, options, onChange }: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (!isOpen) setIsOpen(true);
        else setSelectedIndex((prev) => (prev + 1) % options.length);
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (!isOpen) setIsOpen(true);
        else setSelectedIndex((prev) => (prev - 1 + options.length) % options.length);
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        if (isOpen) {
          onChange(options[selectedIndex].value);
          setIsOpen(false);
          buttonRef.current?.focus();
        } else {
          setIsOpen(true);
        }
        break;
      case 'Escape':
        e.preventDefault();
        setIsOpen(false);
        buttonRef.current?.focus();
        break;
    }
  };

  return (
    <div className="dropdown">
      <button
        ref={buttonRef}
        onClick={() => setIsOpen(!isOpen)}
        onKeyDown={handleKeyDown}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-labelledby="dropdown-label"
      >
        {label}
      </button>

      {isOpen && (
        <ul
          role="listbox"
          aria-labelledby="dropdown-label"
          onKeyDown={handleKeyDown}
          tabIndex={-1}
        >
          {options.map((option, index) => (
            <li
              key={option.value}
              role="option"
              aria-selected={index === selectedIndex}
              onClick={() => { onChange(option.value); setIsOpen(false); }}
            >
              {option.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

### Step 3: Add ARIA Attributes

Provide additional context for screen readers when semantic HTML alone is insufficient.

**Key ARIA attributes:**

| Attribute | Purpose |
|-----------|---------|
| `aria-label` | Directly label an element |
| `aria-labelledby` | Reference another element as a label |
| `aria-describedby` | Provide additional description |
| `aria-live` | Announce dynamic content changes (`polite` / `assertive`) |
| `aria-hidden` | Hide decorative/off-screen elements from screen readers |

**Checklist:**
- [ ] All interactive elements have clear labels
- [ ] Button purpose is clear (e.g., "Submit form" not just "Click")
- [ ] State changes are announced via `aria-live` regions
- [ ] Decorative images use `alt=""` or `aria-hidden="true"`

**Example (Accessible Modal with focus trap):**

```tsx
function AccessibleModal({ isOpen, onClose, title, children }) {
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) modalRef.current?.focus();
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
      aria-describedby="modal-description"
      ref={modalRef}
      tabIndex={-1}
      onKeyDown={(e) => { if (e.key === 'Escape') onClose(); }}
    >
      <div className="modal-overlay" onClick={onClose} aria-hidden="true" />
      <div className="modal-content">
        <h2 id="modal-title">{title}</h2>
        <div id="modal-description">{children}</div>
        <button onClick={onClose} aria-label="Close modal">
          <span aria-hidden="true">×</span>
        </button>
      </div>
    </div>
  );
}
```

**`aria-live` example (notifications):**

```tsx
function Notification({ message, type }: { message: string; type: 'success' | 'error' }) {
  return (
    <div
      role="alert"
      aria-live="assertive"     // Immediate announcement
      aria-atomic="true"        // Read entire content
      className={`notification notification-${type}`}
    >
      {type === 'error' && <span aria-label="Error">⚠️</span>}
      {type === 'success' && <span aria-label="Success">✅</span>}
      {message}
    </div>
  );
}
```

### Step 4: Color Contrast and Visual Accessibility

Ensure sufficient contrast ratios and never rely on color alone to convey information.

**WCAG contrast ratios:**

| Level | Normal text | Large text (18px+ bold or 24px+) |
|-------|-------------|-----------------------------------|
| AA | 4.5:1 | 3:1 |
| AAA | 7:1 | 4.5:1 |

**Example (CSS):**

```css
/* ✅ Sufficient contrast — #0066cc on white = 7.7:1 */
.button {
  background-color: #0066cc;
  color: #ffffff;
}

/* ✅ Visible focus indicator (required) */
button:focus,
a:focus {
  outline: 3px solid #0066cc;
  outline-offset: 2px;
}

/* ❌ outline: none is forbidden — keyboard users lose focus indication */
button:focus {
  outline: none;  /* Never do this */
}

/* ✅ Convey state with color + icon + border */
.error-message {
  color: #d32f2f;
  border-left: 4px solid #d32f2f;
}
.error-message::before {
  content: '⚠️';
  margin-right: 8px;
}
```

### Step 5: Accessibility Testing

**Tasks:**
- Automated scan with axe DevTools
- Check Lighthouse Accessibility score
- Test all features with keyboard only (no mouse)
- Screen reader testing (NVDA, VoiceOver)

**Example (Jest + axe-core):**

```tsx
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

it('should have no accessibility violations', async () => {
  const { container } = render(<AccessibleButton onClick={() => {}}>Click Me</AccessibleButton>);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});

it('should be keyboard accessible', () => {
  const handleClick = jest.fn();
  const { getByRole } = render(<AccessibleButton onClick={handleClick}>Click Me</AccessibleButton>);

  const button = getByRole('button');
  button.focus();
  fireEvent.keyDown(button, { key: 'Enter' });
  expect(handleClick).toHaveBeenCalled();

  fireEvent.keyDown(button, { key: ' ' });
  expect(handleClick).toHaveBeenCalledTimes(2);
});
```

## Output Format: Accessibility Checklist

```markdown
### Semantic HTML
- [ ] Use semantic HTML tags (button, nav, main, etc.)
- [ ] Heading hierarchy is correct (h1 → h2 → h3, no skips)
- [ ] All form labels are connected via for/id or wrapping

### Keyboard Navigation
- [ ] All interactive elements reachable via Tab
- [ ] Buttons activated with Enter/Space
- [ ] Modals/dropdowns closed with ESC
- [ ] Focus indicator is clearly visible (outline)
- [ ] Focus is trapped inside modals/dialogs

### ARIA
- [ ] role attributes used appropriately
- [ ] aria-label or aria-labelledby provided where needed
- [ ] aria-live regions for dynamic content
- [ ] Decorative elements use aria-hidden="true"

### Visual
- [ ] Color contrast meets WCAG AA (4.5:1 normal text)
- [ ] Information not conveyed by color alone
- [ ] Text size can be adjusted (no fixed sizes blocking zoom)

### Testing
- [ ] 0 axe DevTools violations
- [ ] Lighthouse Accessibility score 90+
- [ ] Keyboard-only navigation test passed
- [ ] Screen reader test completed
```

## Constraints

### Required (MUST)
- **Keyboard Accessibility**: All features must be usable without a mouse — support Tab, Enter, Space, arrow keys, and ESC
- **Alternative Text**: All images must have an `alt` attribute — descriptive for meaningful images, `alt=""` for decorative
- **Clear Labels**: All form inputs must have an associated label — `<label for="...">` or `aria-label`; never use placeholder alone as a substitute

### Prohibited (MUST NOT)
- **Do Not Remove Outline**: Never use `outline: none` without providing a custom focus style — it destroys keyboard navigation
- **Do Not Use tabindex > 0**: Avoid changing natural focus order — keep DOM order logical
- **Do Not Convey Information by Color Alone**: Always accompany with icons, text, or patterns — consider color blindness

## Best Practices

- **Semantic HTML First**: ARIA is a last resort — the correct HTML element (`<button>`, `<nav>`, etc.) often makes ARIA unnecessary
- **Focus Management**: In SPAs, move focus to main content on page transitions; provide skip links ("Skip to main content")
- **Error Messages**: Be specific — "Invalid input" → "Email must be in format: example@domain.com"

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Removing `outline` on focus | Always provide a custom focus indicator with `outline` or `box-shadow` |
| Using `<div>` as a button | Use `<button>` — it's free with keyboard support and ARIA roles |
| Placeholder as only label | Placeholders disappear on input; always use `<label>` |
| Low color contrast | Test with contrast checkers; aim for 4.5:1 minimum |
| No focus trap in modals | Trap Tab cycling inside open modals; return focus on close |
| `aria-live="assertive"` for everything | Use `polite` for non-critical updates; reserve `assertive` for errors/alerts |

## Examples

### Accessible Form

```tsx
function AccessibleContactForm() {
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitStatus, setSubmitStatus] = useState<'idle' | 'success' | 'error'>('idle');

  return (
    <form onSubmit={handleSubmit} noValidate>
      <h2 id="form-title">Contact Us</h2>
      <p id="form-description">Please fill out the form below to get in touch.</p>

      <div className="form-group">
        <label htmlFor="name">Name <span aria-label="required">*</span></label>
        <input
          type="text" id="name" name="name" required
          aria-required="true"
          aria-invalid={!!errors.name}
          aria-describedby={errors.name ? 'name-error' : undefined}
        />
        {errors.name && (
          <span id="name-error" role="alert" className="error">{errors.name}</span>
        )}
      </div>

      <div className="form-group">
        <label htmlFor="email">Email <span aria-label="required">*</span></label>
        <input
          type="email" id="email" name="email" required
          aria-required="true"
          aria-invalid={!!errors.email}
          aria-describedby={errors.email ? 'email-error' : 'email-hint'}
        />
        <span id="email-hint" className="hint">We'll never share your email.</span>
        {errors.email && (
          <span id="email-error" role="alert" className="error">{errors.email}</span>
        )}
      </div>

      <button type="submit" disabled={submitStatus === 'loading'}>
        {submitStatus === 'loading' ? 'Submitting...' : 'Submit'}
      </button>

      {submitStatus === 'success' && (
        <div role="alert" aria-live="polite" className="success">✅ Form submitted!</div>
      )}
      {submitStatus === 'error' && (
        <div role="alert" aria-live="assertive" className="error">⚠️ Error. Please try again.</div>
      )}
    </form>
  );
}
```

### Accessible Tabs

```tsx
function AccessibleTabs({ tabs }: { tabs: { id: string; label: string; content: ReactNode }[] }) {
  const [activeTab, setActiveTab] = useState(0);

  const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
    switch (e.key) {
      case 'ArrowRight': e.preventDefault(); setActiveTab((index + 1) % tabs.length); break;
      case 'ArrowLeft': e.preventDefault(); setActiveTab((index - 1 + tabs.length) % tabs.length); break;
      case 'Home': e.preventDefault(); setActiveTab(0); break;
      case 'End': e.preventDefault(); setActiveTab(tabs.length - 1); break;
    }
  };

  return (
    <div>
      <div role="tablist" aria-label="Content sections">
        {tabs.map((tab, index) => (
          <button
            key={tab.id}
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={activeTab === index}
            aria-controls={`panel-${tab.id}`}
            tabIndex={activeTab === index ? 0 : -1}
            onClick={() => setActiveTab(index)}
            onKeyDown={(e) => handleKeyDown(e, index)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {tabs.map((tab, index) => (
        <div
          key={tab.id}
          role="tabpanel"
          id={`panel-${tab.id}`}
          aria-labelledby={`tab-${tab.id}`}
          hidden={activeTab !== index}
          tabIndex={0}
        >
          {tab.content}
        </div>
      ))}
    </div>
  );
}
```

## Related Skills

- **responsive-design**: Ensure responsive layouts remain keyboard- and screen-reader-friendly
- **code-review**: Audit code for accessibility violations during review

## References

- [WCAG 2.1 Guidelines](https://www.w3.org/TR/WCAG21/)
- [MDN ARIA](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA)
- [WebAIM](https://webaim.org/)
- [axe DevTools](https://www.deque.com/axe/)
- [A11y Project](https://www.a11yproject.com/)
