---
name: responsive-design
description: "Use when building new websites/apps with mobile-first layouts, converting fixed layouts to responsive, optimizing images per device, or supporting tablet/desktop/large screens - covers Flexbox, CSS Grid, media queries, responsive images, fluid typography with clamp(), and container queries."
---

# Responsive Design

## Overview

Mobile-first CSS layouts using Flexbox, Grid, media queries, responsive images, fluid typography, and container queries — covering 320px through 1440px+ screens.

## When to Use

- **New website/app**: Layout design for combined mobile-desktop use
- **Legacy improvement**: Converting fixed layouts to responsive
- **Performance optimization**: Image optimization per device
- **Multiple screens**: Tablet, desktop, and large screen support

## Core Workflow

### Step 1: Mobile-First Approach

Design from small screens and progressively expand with `min-width` media queries.

```css
/* Default: Mobile (320px~) */
.container {
  padding: 1rem;
  font-size: 14px;
}

.grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
}

/* Tablet (768px~) */
@media (min-width: 768px) {
  .container {
    padding: 2rem;
    font-size: 16px;
  }
  .grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
  }
}

/* Desktop (1024px~) */
@media (min-width: 1024px) {
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 3rem;
  }
  .grid {
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
  }
}

/* Large screen (1440px~) */
@media (min-width: 1440px) {
  .grid {
    grid-template-columns: repeat(4, 1fr);
  }
}
```

### Step 2: Flexbox / Grid Layout

**Flexbox (1-dimensional layout):**

```css
/* Navigation bar */
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
}

/* Card list */
.card-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

@media (min-width: 768px) {
  .card-list {
    flex-direction: row;
    flex-wrap: wrap;
  }
  .card {
    flex: 1 1 calc(50% - 0.5rem);
  }
}

@media (min-width: 1024px) {
  .card {
    flex: 1 1 calc(33.333% - 0.667rem);
  }
}
```

**CSS Grid (2-dimensional layout):**

```css
.dashboard {
  display: grid;
  grid-template-areas:
    "header"
    "sidebar"
    "main"
    "footer";
  gap: 1rem;
}

@media (min-width: 768px) {
  .dashboard {
    grid-template-areas:
      "header header"
      "sidebar main"
      "footer footer";
    grid-template-columns: 250px 1fr;
  }
}

@media (min-width: 1024px) {
  .dashboard {
    grid-template-columns: 300px 1fr;
  }
}

.header { grid-area: header; }
.sidebar { grid-area: sidebar; }
.main { grid-area: main; }
.footer { grid-area: footer; }
```

### Step 3: Responsive Images

**Resolution switching with `srcset`:**

```html
<img
  src="image-800.jpg"
  srcset="
    image-400.jpg 400w,
    image-800.jpg 800w,
    image-1200.jpg 1200w,
    image-1600.jpg 1600w
  "
  sizes="
    (max-width: 600px) 100vw,
    (max-width: 900px) 50vw,
    33vw
  "
  alt="Responsive image"
/>
```

**Art direction with `<picture>`:**

```html
<picture>
  <source media="(max-width: 767px)" srcset="portrait.jpg">
  <source media="(max-width: 1023px)" srcset="square.jpg">
  <img src="landscape.jpg" alt="Art direction example">
</picture>
```

**CSS background images:**

```css
.hero {
  background-image: url('hero-mobile.jpg');
}

@media (min-width: 768px) {
  .hero { background-image: url('hero-tablet.jpg'); }
}

@media (min-width: 1024px) {
  .hero { background-image: url('hero-desktop.jpg'); }
}

/* Or use image-set() */
.hero {
  background-image: image-set(
    url('hero-1x.jpg') 1x,
    url('hero-2x.jpg') 2x
  );
}
```

### Step 4: Responsive Typography

**Fluid sizing with `clamp()`:**

```css
:root {
  --font-size-body: clamp(14px, 2.5vw, 18px);
  --font-size-h1: clamp(24px, 5vw, 48px);
  --font-size-h2: clamp(20px, 4vw, 36px);
}

body { font-size: var(--font-size-body); }
h1 { font-size: var(--font-size-h1); line-height: 1.2; }
h2 { font-size: var(--font-size-h2); line-height: 1.3; }
```

**Media query approach:**

```css
body { font-size: 14px; line-height: 1.6; }

@media (min-width: 768px) { body { font-size: 16px; } }
@media (min-width: 1024px) { body { font-size: 18px; } }
```

### Step 5: Container Queries

Apply styles based on parent container size, not viewport.

```css
.card-container {
  container-type: inline-size;
  container-name: card;
}

.card { padding: 1rem; }
.card h2 { font-size: 1.2rem; }

@container card (min-width: 400px) {
  .card {
    display: grid;
    grid-template-columns: 200px 1fr;
    padding: 1.5rem;
  }
  .card h2 { font-size: 1.5rem; }
}

@container card (min-width: 600px) {
  .card {
    grid-template-columns: 300px 1fr;
    padding: 2rem;
  }
}
```

## Standard Breakpoints

```css
/* Mobile (default): 320px ~ 767px */
/* Tablet: 768px ~ 1023px */
/* Desktop: 1024px ~ 1439px */
/* Large: 1440px+ */

:root {
  --breakpoint-sm: 640px;
  --breakpoint-md: 768px;
  --breakpoint-lg: 1024px;
  --breakpoint-xl: 1280px;
  --breakpoint-2xl: 1536px;
}
```

## Constraints

### Required (MUST)
- **Viewport meta tag**: Always include `<meta name="viewport" content="width=device-width, initial-scale=1.0">`
- **Mobile-First**: Mobile default, use `min-width` media queries only — never `max-width` for layout
- **Relative units**: Use `rem` (font-size), `rem`/`em` (padding/margin), `%`/`vw` (width)

### Prohibited (MUST NOT)
- **Fixed widths**: Never `width: 1200px` — use `max-width: 1200px`
- **Duplicate code**: Common styles as default, only differences in media queries

## Best Practices

- **Container queries first**: Prefer `@container` over `@media` when the component's parent size is the controlling factor
- **Flexbox vs Grid**: Flexbox for 1-dimensional (nav, card rows), Grid for 2-dimensional (dashboards, page layouts)
- **Performance**: Lazy-load images with `loading="lazy"`, use WebP/AVIF with `<picture>` fallbacks
- **Testing**: Chrome DevTools Device Mode, BrowserStack for real device testing

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Desktop-first with `max-width` queries | Mobile-first with `min-width` queries |
| Fixed pixel widths for containers | `max-width` + `%` or `vw` units |
| Ignoring viewport meta tag | Always include it in `<head>` |
| Images that don't scale | `max-width: 100%; height: auto` + `srcset` |
| Huge font sizes on mobile | `clamp()` with a viewport-relative preferred value |
| Forgetting `gap` support | Use `gap` for consistent spacing (supported everywhere now) |

## Examples

### Responsive Navigation

```jsx
function ResponsiveNav() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="navbar">
      <a href="/" className="logo">MyApp</a>
      <button
        className="menu-toggle"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle menu"
        aria-expanded={isOpen}
      >
        <span></span><span></span><span></span>
      </button>
      <ul className={`nav-links ${isOpen ? 'active' : ''}`}>
        <li><a href="/about">About</a></li>
        <li><a href="/services">Services</a></li>
        <li><a href="/contact">Contact</a></li>
      </ul>
    </nav>
  );
}
```

```css
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
}
.nav-links {
  display: none;
  position: absolute;
  top: 60px;
  left: 0;
  right: 0;
  background: white;
  flex-direction: column;
}
.nav-links.active { display: flex; }

@media (min-width: 768px) {
  .menu-toggle { display: none; }
  .nav-links {
    display: flex;
    position: static;
    flex-direction: row;
    gap: 2rem;
  }
}
```

### Responsive Product Grid

```jsx
function ProductGrid({ products }) {
  return (
    <div className="product-grid">
      {products.map(product => (
        <div key={product.id} className="product-card">
          <img src={product.image} alt={product.name} />
          <h3>{product.name}</h3>
          <p>${product.price}</p>
          <button>Add to Cart</button>
        </div>
      ))}
    </div>
  );
}
```

```css
.product-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
  padding: 1rem;
}
.product-card img {
  width: 100%;
  height: auto;
  aspect-ratio: 1;
  object-fit: cover;
}

@media (min-width: 640px) { .product-grid { grid-template-columns: repeat(2, 1fr); } }
@media (min-width: 1024px) { .product-grid { grid-template-columns: repeat(3, 1fr); gap: 1.5rem; } }
@media (min-width: 1440px) { .product-grid { grid-template-columns: repeat(4, 1fr); gap: 2rem; } }
```

## Related Skills

- **accessibility**: Ensure responsive layouts remain keyboard- and screen-reader-friendly
- **impeccable**: Visual polish for responsive interfaces

## References

- [MDN Responsive Design](https://developer.mozilla.org/en-US/docs/Learn/CSS/CSS_layout/Responsive_Design)
- [CSS Grid Guide (CSS-Tricks)](https://css-tricks.com/snippets/css/complete-guide-grid/)
- [Flexbox Guide (CSS-Tricks)](https://css-tricks.com/snippets/css/a-guide-to-flexbox/)
- [Container Queries (MDN)](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_container_queries)
