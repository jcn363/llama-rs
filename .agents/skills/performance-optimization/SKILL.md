---
name: performance-optimization
description: "Use when diagnosing slow page loads, low Lighthouse scores, delayed user interactions, large bundle sizes, or slow database queries - provides frontend optimization (React.memo, useMemo, lazy loading, code splitting, image optimization, bundle analysis) and backend strategies (N+1 fix, indexing, Redis caching, API compression)."
---

# Performance Optimization

## Overview

Diagnose and fix performance bottlenecks across React frontends and database backends — profiling first, then incremental improvements with continuous monitoring.

## When to Use

- **Slow page loads**: Low Lighthouse score
- **Slow rendering**: Delayed user interactions
- **Large bundle size**: Increased download time
- **Slow queries**: Database bottlenecks

## Core Workflow

### Step 1: Measure Performance

**Lighthouse (Chrome DevTools):**

```bash
# CLI
npm install -g lighthouse
lighthouse https://example.com --view

# Automate in CI
lighthouse https://example.com --output=json --output-path=./report.json
```

**Measure Web Vitals (React):**

```typescript
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

function sendToAnalytics(metric: any) {
  console.log(metric); // Send to GA, Datadog, etc.
}

getCLS(sendToAnalytics);
getFID(sendToAnalytics);
getFCP(sendToAnalytics);
getLCP(sendToAnalytics);
getTTFB(sendToAnalytics);
```

### Step 2: Optimize React

**React.memo (prevent unnecessary re-renders):**

```tsx
// ❌ Bad: child re-renders whenever parent re-renders
function ExpensiveComponent({ data }: { data: Data }) {
  return <div>{/* complex rendering */}</div>;
}

// ✅ Good: re-render only when props change
const ExpensiveComponent = React.memo(({ data }: { data: Data }) => {
  return <div>{/* complex rendering */}</div>;
});
```

**useMemo & useCallback:**

```tsx
function ProductList({ products, category }: Props) {
  // ✅ Memoize filtered results
  const filteredProducts = useMemo(() => {
    return products.filter(p => p.category === category);
  }, [products, category]);

  // ✅ Memoize callback
  const handleAddToCart = useCallback((id: string) => {
    addToCart(id);
  }, []);

  return (
    <div>
      {filteredProducts.map(product => (
        <ProductCard key={product.id} product={product} onAdd={handleAddToCart} />
      ))}
    </div>
  );
}
```

**Lazy Loading & Code Splitting:**

```tsx
import { lazy, Suspense } from 'react';

// ✅ Route-based code splitting
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Profile = lazy(() => import('./pages/Profile'));
const Settings = lazy(() => import('./pages/Settings'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Suspense>
  );
}

// ✅ Component-based lazy loading
const HeavyChart = lazy(() => import('./components/HeavyChart'));

function Dashboard() {
  return (
    <div>
      <h1>Dashboard</h1>
      <Suspense fallback={<Skeleton />}>
        <HeavyChart data={data} />
      </Suspense>
    </div>
  );
}
```

### Step 3: Optimize Bundle Size

**Webpack Bundle Analyzer:**

```bash
npm install --save-dev webpack-bundle-analyzer
```

```json
{
  "scripts": {
    "analyze": "webpack-bundle-analyzer build/stats.json"
  }
}
```

**Tree Shaking — import only what you need:**

```typescript
// ❌ Bad: imports entire library
import _ from 'lodash';

// ✅ Good: import only what you need
import debounce from 'lodash/debounce';
```

**Dynamic Imports:**

```typescript
// ✅ Load only when needed
button.addEventListener('click', async () => {
  const { default: Chart } = await import('chart.js');
  new Chart(ctx, config);
});
```

### Step 4: Optimize Images

**Next.js Image component:**

```tsx
import Image from 'next/image';

function ProductImage() {
  return (
    <Image
      src="/product.jpg"
      alt="Product"
      width={500}
      height={500}
      priority        // for the LCP image
      placeholder="blur"  // blur placeholder
      sizes="(max-width: 768px) 100vw, 50vw"
    />
  );
}
```

**Use WebP format:**

```html
<picture>
  <source srcset="image.webp" type="image/webp">
  <source srcset="image.jpg" type="image/jpeg">
  <img src="image.jpg" alt="Fallback">
</picture>
```

### Step 5: Optimize Database Queries

**Fix the N+1 query problem:**

```typescript
// ❌ Bad: N+1 queries
const posts = await db.post.findMany();
for (const post of posts) {
  const author = await db.user.findUnique({ where: { id: post.authorId } });
  // 101 queries (1 + 100)
}

// ✅ Good: JOIN or include
const posts = await db.post.findMany({
  include: { author: true }
});
// 1 query
```

**Add indexes:**

```sql
-- Identify slow queries
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- Add index
CREATE INDEX idx_users_email ON users(email);

-- Composite index
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);
```

**Caching (Redis):**

```typescript
async function getUserProfile(userId: string) {
  // 1. Check cache
  const cached = await redis.get(`user:${userId}`);
  if (cached) return JSON.parse(cached);

  // 2. Query DB
  const user = await db.user.findUnique({ where: { id: userId } });

  // 3. Store in cache (1 hour TTL)
  await redis.setex(`user:${userId}`, 3600, JSON.stringify(user));

  return user;
}
```

## Output Format

```markdown
## Performance Optimization Checklist

### Frontend
- [ ] Prevent unnecessary re-renders with React.memo
- [ ] Use useMemo/useCallback appropriately
- [ ] Lazy loading & Code splitting
- [ ] Optimize images (WebP, lazy loading)
- [ ] Analyze and reduce bundle size

### Backend
- [ ] Remove N+1 queries
- [ ] Add database indexes
- [ ] Redis caching
- [ ] Compress API responses (gzip)
- [ ] Use a CDN

### Measurement
- [ ] Lighthouse score 90+
- [ ] LCP < 2.5s
- [ ] FID < 100ms
- [ ] CLS < 0.1
```

## Constraints

### Required (MUST)
- **Measure first**: Profile, don't guess — always benchmark before and after
- **Incremental improvements**: Optimize one thing at a time to isolate impact
- **Performance monitoring**: Track continuously in production

### Prohibited (MUST NOT)
- **Premature optimization**: Don't optimize without identified bottlenecks
- **Sacrificing readability**: Don't make code complex for marginal perf gains

## Best Practices

- **80/20 rule**: 80% of improvement comes from 20% of effort — find the low-hanging fruit first
- **User-centered**: Focus on metrics that impact real user experience (LCP, FID, CLS)
- **Automation**: Add performance regression tests in CI

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Optimizing without profiling | Measure first — the bottleneck is rarely where you expect |
| Over-using React.memo/useMemo | Profile to confirm it helps; unnecessary memoization costs memory |
| Forgetting images are the #1 perf killer | Always lazy-load below-fold images, use WebP/AVIF |
| Ignoring the N+1 problem | Use `EXPLAIN ANALYZE` and look for repeated identical queries |
| Caching everything | Cache only expensive/stable data; stale caches cause bugs |

## Related Skills

- **responsive-design**: Responsive image optimization and layout performance
- **data-analysis**: Query performance analysis patterns

## References

- [web.dev/vitals](https://web.dev/vitals/)
- [React Optimization (React docs)](https://react.dev/reference/react/memo)
- [Webpack Bundle Analyzer](https://github.com/webpack-contrib/webpack-bundle-analyzer)
- [Use EXPLAIN ANALYZE (PostgreSQL)](https://www.postgresql.org/docs/current/using-explain.html)
