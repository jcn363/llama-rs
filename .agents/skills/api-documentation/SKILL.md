---
name: api-documentation
description: "Use when adding documentation for new API endpoints, preparing a public API launch, defining frontend-backend interfaces, generating OpenAPI/Swagger specs, or setting up interactive API docs - covers OpenAPI 3.0 spec generation, JSDoc/decorator-based extraction, Swagger UI integration, authentication docs, and multi-language code examples."
---

# API Documentation

## Overview

Generate comprehensive API documentation from OpenAPI specs, code comments (JSDoc/decorators), and interactive examples — covering schemas, security schemes, error codes, rate limiting, pagination, and multi-language code examples.

## When to Use

- **API Development**: When adding new endpoints
- **External Release**: Public API launch
- **Team Collaboration**: Frontend-backend interface definition

## Core Workflow

### Step 1: OpenAPI (Swagger) Spec

```yaml
openapi: 3.0.0
info:
  title: User Management API
  version: 1.0.0
  description: API for managing users
  contact:
    email: api@example.com

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://staging-api.example.com/v1
    description: Staging

paths:
  /users:
    get:
      summary: List all users
      description: Retrieve a paginated list of users
      tags:
        - Users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  pagination:
                    $ref: '#/components/schemas/Pagination'
        '401':
          $ref: '#/components/responses/Unauthorized'

    post:
      summary: Create a new user
      tags:
        - Users
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/BadRequest'

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          format: uuid
        email:
          type: string
          format: email
        name:
          type: string
        createdAt:
          type: string
          format: date-time
      required:
        - id
        - email
        - name

    CreateUserRequest:
      type: object
      properties:
        email:
          type: string
          format: email
        name:
          type: string
          minLength: 2
          maxLength: 50
        password:
          type: string
          minLength: 8
      required:
        - email
        - name
        - password

    Pagination:
      type: object
      properties:
        page:
          type: integer
        limit:
          type: integer
        total:
          type: integer

  responses:
    Unauthorized:
      description: Unauthorized
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                type: string
                example: "Authentication required"

    BadRequest:
      description: Bad Request
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                type: string
                example: "Invalid input"

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - bearerAuth: []
```

### Step 2: Generate Documentation from Code (JSDoc/Decorators)

Use JSDoc annotations with swagger-jsdoc to auto-generate the spec from Express routes:

```typescript
/**
 * @swagger
 * /users:
 *   post:
 *     summary: Create a new user
 *     tags: [Users]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - email
 *               - name
 *               - password
 *             properties:
 *               email:
 *                 type: string
 *                 format: email
 *               name:
 *                 type: string
 *               password:
 *                 type: string
 *                 minLength: 8
 *     responses:
 *       201:
 *         description: User created successfully
 *       400:
 *         description: Invalid input
 */
router.post('/users', async (req, res) => {
  const { email, name, password } = req.body;
  const user = await userService.createUser({ email, name, password });
  res.status(201).json(user);
});
```

### Step 3: Interactive Documentation (Swagger UI)

```typescript
import swaggerUi from 'swagger-ui-express';
import YAML from 'yamljs';

const swaggerDocument = YAML.load('./openapi.yaml');

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument, {
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: "My API Documentation"
}));
```

### Step 4: Examples & Guides

```
## API Documentation

### Authentication

All API requests require authentication using JWT tokens.

#### Getting a Token
\`\`\`bash
curl -X POST https://api.example.com/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "yourpassword"}'
\`\`\`

Response:
\`\`\`json
{
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refreshToken": "..."
}
\`\`\`

#### Using the Token
\`\`\`bash
curl -X GET https://api.example.com/v1/users \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
\`\`\`

### Creating a User

**Endpoint**: `POST /v1/users`

**Request Body**:
\`\`\`json
{
  "email": "john@example.com",
  "name": "John Doe",
  "password": "SecurePass123!"
}
\`\`\`

**Success Response** (201):
\`\`\`json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "email": "john@example.com",
  "name": "John Doe",
  "createdAt": "2025-01-15T10:00:00Z"
}
\`\`\`

**Error Response** (400):
\`\`\`json
{
  "error": "Email already exists"
}
\`\`\`

### Rate Limiting
- 100 requests per 15 minutes per IP
- Header: `X-RateLimit-Remaining`

### Pagination
\`\`\`
GET /v1/users?page=2&limit=20
\`\`\`

Response includes pagination info:
\`\`\`json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 20,
    "total": 157,
    "pages": 8
  }
}
\`\`\`

### Error Codes
- `400` — Bad Request (validation error)
- `401` — Unauthorized (missing/invalid token)
- `403` — Forbidden (insufficient permissions)
- `404` — Not Found
- `409` — Conflict (duplicate resource)
- `429` — Too Many Requests (rate limit)
- `500` — Internal Server Error
```

## Output Format: Documentation Structure

```
docs/
├── README.md                    # Overview
├── getting-started.md           # Quick start guide
├── authentication.md            # Auth guide
├── api-reference/
│   ├── users.md                 # Users endpoints
│   ├── auth.md                  # Auth endpoints
│   └── products.md              # Products endpoints
├── guides/
│   ├── pagination.md
│   ├── error-handling.md
│   └── rate-limiting.md
├── examples/
│   ├── curl.md
│   ├── javascript.md
│   └── python.md
└── openapi.yaml                 # OpenAPI spec
```

## Constraints

### Required (MUST)
- **Real Examples**: Provide working code examples (curl, JS, Python)
- **Error Cases**: Document not only success but also failure cases and error codes
- **Keep Updated**: Update documentation when API changes — automate with spec generation from code

### Prohibited (MUST NOT)
- **Real Keys in Examples**: Never use real API keys, passwords, or tokens in examples
- **Vague Descriptions**: Avoid unclear descriptions like "returns data" — be specific

## Best Practices

- **Try It Out**: Provide interactive documentation (Swagger UI) so consumers can test endpoints directly
- **Provide SDKs**: Offer code examples in major languages (curl, JavaScript, Python, Go, etc.)
- **Changelog**: Document API changes and version history in a changelog
- **Consistent naming**: Use the same parameter names, error formats, and casing everywhere

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Only documenting success paths | Always include error responses and status codes |
| Stale docs that don't match the code | Generate specs from code annotations (JSDoc/decorators) |
| No auth examples | Show exact header format and token acquisition flow |
| Real secrets in examples | Use placeholder values like `YOUR_API_KEY` |
| One monolithic doc page | Organize into reference, guides, and examples |

## Related Skills

- **api-design**: REST resource naming, status codes, and versioning strategies
- **security-best-practices**: Authentication and token management patterns

## References

- [OpenAPI Specification](https://spec.openapis.org/oas/v3.0.3)
- [Swagger UI](https://swagger.io/tools/swagger-ui/)
- [Redoc](https://redocly.com/redoc)
