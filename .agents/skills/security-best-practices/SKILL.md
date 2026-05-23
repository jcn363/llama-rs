---
name: security-best-practices
description: "Use when hardening web application security, auditing for OWASP Top 10 vulnerabilities, setting up HTTPS/security headers, preventing SQL Injection/XSS/CSRF, implementing JWT authentication with refresh token rotation, or complying with GDPR/PCI-DSS - provides comprehensive security middleware, input validation, secret management, and authentication patterns."
---

# Security Best Practices

## Overview

Comprehensive security hardening covering HTTPS enforcement, security headers, input validation, SQL Injection/XSS/CSRF prevention, JWT authentication with refresh token rotation, secret management, and OWASP Top 10 compliance.

## When to Use

- **New project**: Consider security from the start
- **Security audit**: Inspect and fix vulnerabilities
- **Public API**: Harden APIs accessible externally
- **Compliance**: Comply with GDPR, PCI-DSS, etc.

## Core Workflow

### Step 1: Enforce HTTPS and Security Headers

Use Helmet and rate limiting middleware:

```javascript
import express from 'express';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

const app = express();

// Helmet: automatically set security headers
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'", "https://trusted-cdn.com"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "https://api.example.com"],
      fontSrc: ["'self'", "https:", "data:"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  }
}));

// Enforce HTTPS
app.use((req, res, next) => {
  if (process.env.NODE_ENV === 'production' && !req.secure) {
    return res.redirect(301, `https://${req.headers.host}${req.url}`);
  }
  next();
});

// Rate limiting (DDoS prevention)
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // max 100 requests per IP
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});

app.use('/api/', limiter);

// Stricter for auth endpoints
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5, // only 5 attempts per 15 minutes
  skipSuccessfulRequests: true // do not count successful requests
});

app.use('/api/auth/login', authLimiter);
```

### Step 2: Input Validation (SQL Injection, XSS Prevention)

Validate all user input with a schema library and use parameterized queries:

```javascript
import Joi from 'joi';

const userSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string()
    .min(8)
    .pattern(/^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/)
    .required(),
  name: Joi.string().min(2).max(50).required()
});

app.post('/api/users', async (req, res) => {
  // 1. Validate input
  const { error, value } = userSchema.validate(req.body);
  if (error) {
    return res.status(400).json({ error: error.details[0].message });
  }

  // 2. Prevent SQL Injection: Parameterized Queries
  // ❌ Bad: db.query(`SELECT * FROM users WHERE email = '${email}'`);
  // ✅ Good:
  const user = await db.query('SELECT * FROM users WHERE email = ?', [value.email]);

  // 3. Prevent XSS: Output Encoding
  // React/Vue escape automatically; otherwise use a library
  import DOMPurify from 'isomorphic-dompurify';
  const sanitized = DOMPurify.sanitize(value.name);

  res.json({ user: sanitized });
});
```

### Step 3: Prevent CSRF

Use CSRF tokens on state-changing requests:

```javascript
import csrf from 'csurf';
import cookieParser from 'cookie-parser';

app.use(cookieParser());

// CSRF protection
const csrfProtection = csrf({ cookie: true });

// Provide CSRF token
app.get('/api/csrf-token', csrfProtection, (req, res) => {
  res.json({ csrfToken: req.csrfToken() });
});

// Validate CSRF on all POST/PUT/DELETE requests
app.post('/api/*', csrfProtection, (req, res, next) => {
  next();
});

// Client usage:
// fetch('/api/users', {
//   method: 'POST',
//   headers: { 'CSRF-Token': csrfToken },
//   body: JSON.stringify(data)
// });
```

### Step 4: Manage Secrets

**Never commit secrets to version control.**

`.env` file (add to `.gitignore`):

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mydb

# JWT
ACCESS_TOKEN_SECRET=your-super-secret-access-token-key-min-32-chars
REFRESH_TOKEN_SECRET=your-super-secret-refresh-token-key-min-32-chars

# API Keys
STRIPE_SECRET_KEY=sk_test_xxx
SENDGRID_API_KEY=SG.xxx
```

**Kubernetes Secrets:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: myapp-secrets
type: Opaque
stringData:
  database-url: postgresql://user:password@postgres:5432/mydb
  jwt-secret: your-jwt-secret
```

Always read from environment variables and fail fast if missing:

```javascript
const dbUrl = process.env.DATABASE_URL;
if (!dbUrl) {
  throw new Error('DATABASE_URL environment variable is required');
}
```

### Step 5: Secure API Authentication (JWT + Refresh Token Rotation)

Use short-lived access tokens with refresh token rotation to limit breach impact:

```javascript
// Short-lived access token (15 minutes)
const accessToken = jwt.sign({ userId }, ACCESS_SECRET, { expiresIn: '15m' });

// Long-lived refresh token (7 days), store in DB
const refreshToken = jwt.sign({ userId }, REFRESH_SECRET, { expiresIn: '7d' });
await db.refreshToken.create({
  userId,
  token: refreshToken,
  expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
});

// Refresh token rotation: re-issue on each use
app.post('/api/auth/refresh', async (req, res) => {
  const { refreshToken } = req.body;
  const payload = jwt.verify(refreshToken, REFRESH_SECRET);

  // Invalidate existing token (rotation prevents replay)
  await db.refreshToken.delete({ where: { token: refreshToken } });

  // Issue new tokens
  const newAccessToken = jwt.sign(
    { userId: payload.userId }, ACCESS_SECRET, { expiresIn: '15m' }
  );
  const newRefreshToken = jwt.sign(
    { userId: payload.userId }, REFRESH_SECRET, { expiresIn: '7d' }
  );

  await db.refreshToken.create({
    userId: payload.userId,
    token: newRefreshToken,
    expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
  });

  res.json({ accessToken: newAccessToken, refreshToken: newRefreshToken });
});
```

## OWASP Top 10 Checklist

- [ ] **A01: Broken Access Control** — RBAC, authorization checks on every endpoint
- [ ] **A02: Cryptographic Failures** — HTTPS, strong encryption at rest and in transit
- [ ] **A03: Injection** — Parameterized queries, input validation
- [ ] **A04: Insecure Design** — Security by design, threat modeling
- [ ] **A05: Security Misconfiguration** — Helmet, change default passwords, disable directory listing
- [ ] **A06: Vulnerable Components** — `npm audit`, regular dependency updates
- [ ] **A07: Authentication Failures** — Strong auth, MFA, rate-limit login attempts
- [ ] **A08: Data Integrity Failures** — Signature validation, CSRF prevention
- [ ] **A09: Logging Failures** — Security event logging, monitoring, and alerting
- [ ] **A10: SSRF** — Validate and restrict outbound requests

## Constraints

### Required (MUST)
- **HTTPS Only**: HTTPS required in production
- **Separate secrets**: Manage via environment variables; never hardcode in code
- **Input Validation**: Validate all user input server-side
- **Parameterized Queries**: Prevent SQL Injection — never interpolate user input into SQL
- **Rate Limiting**: DDoS prevention on public endpoints

### Prohibited (MUST NOT)
- No `eval()` — code injection risk
- No direct `innerHTML` — XSS risk
- No committing secrets — never commit `.env` files or credentials to version control
- No rolling your own crypto — use well-audited libraries

## Best Practices

- **Principle of Least Privilege**: Grant minimal necessary permissions (DB, IAM, file system)
- **Defense in Depth**: Layer security controls — no single point of failure
- **Security Audits**: Regular security reviews, dependency scanning, and penetration testing
- **Fail Securely**: Default-deny access; errors should not leak sensitive information

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Storing passwords in plaintext | Use bcrypt/argon2 for hashing |
| Trusting user input blindly | Validate and sanitize everything |
| Only validating on the client | Always validate server-side |
| Using `eval()` or `new Function()` | Use safe parsers or sandboxes |
| Committing `.env` files | Add to `.gitignore`; use vault/CI secrets |
| Long-lived JWT tokens (days) | Short access tokens (15m) + refresh rotation |
| Missing rate limiting on auth | Apply strict rate limits to login endpoints |

## Related Skills

- **task-planning**: Sprint planning for security work items

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [helmet.js](https://helmetjs.github.io/)
- [Security Checklist (Snyk)](https://snyk.io/blog/security-checklist/)
