---
name: api-design
description: "Use when designing new REST APIs, creating GraphQL schemas, refactoring API endpoints, documenting API specifications with OpenAPI, planning versioning strategies, or defining data models and relationships - covers REST fundamentals, pagination/filtering/sorting, authentication (JWT/OAuth 2.0/API keys), URL and header versioning, OpenAPI 3.0 templates, and GraphQL alternatives."
---

# API Design

## Overview

REST and GraphQL API design covering resource naming, HTTP methods, status codes, pagination, filtering, sorting, authentication, versioning, OpenAPI 3.0 documentation, and GraphQL schema design.

## When to Use

- Designing new REST APIs
- Creating GraphQL schemas
- Refactoring API endpoints
- Documenting API specifications
- API versioning strategies
- Defining data models and relationships

## Core Workflow

### Step 1: Define API Requirements

1. Identify resources and entities
2. Define relationships between entities
3. Specify operations (CRUD, custom actions)
4. Plan authentication/authorization
5. Consider pagination, filtering, sorting

### Step 2: Design REST API

**Resource naming:**
- Use nouns, not verbs: `/users` not `/getUsers`
- Use plural names: `/users/{id}`
- Nest resources logically: `/users/{id}/posts`
- Keep URLs short and intuitive

**HTTP methods:**

| Method | Purpose | Idempotent |
|--------|---------|------------|
| `GET` | Retrieve resources | Yes |
| `POST` | Create new resources | No |
| `PUT` | Replace entire resource | Yes |
| `PATCH` | Partial update | No |
| `DELETE` | Remove resources | Yes |

**Status codes:**

| Code | Meaning |
|------|---------|
| `200 OK` | Success with response body |
| `201 Created` | Resource created |
| `204 No Content` | Success, no body |
| `400 Bad Request` | Invalid input |
| `401 Unauthorized` | Authentication required |
| `403 Forbidden` | No permission |
| `404 Not Found` | Resource doesn't exist |
| `409 Conflict` | Resource conflict |
| `422 Unprocessable Entity` | Validation failed |
| `500 Internal Server Error` | Server error |

**Example REST endpoints:**

```
GET    /api/v1/users           # List users
GET    /api/v1/users/{id}      # Get user
POST   /api/v1/users           # Create user
PUT    /api/v1/users/{id}      # Update user
PATCH  /api/v1/users/{id}      # Partial update
DELETE /api/v1/users/{id}      # Delete user
```

### Step 3: Request/Response Format

**Request:**

```http
POST /api/v1/users
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "role": "admin"
}
```

**Response:**

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /api/v1/users/123

{
  "id": 123,
  "name": "John Doe",
  "email": "john@example.com",
  "role": "admin",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### Step 4: Error Handling

Standard error response envelope:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input provided",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  }
}
```

### Step 5: Pagination

**Query parameters:**

```
GET /api/v1/users?page=2&limit=20&sort=-created_at&filter=role:admin
```

**Response with pagination envelope:**

```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 20,
    "total": 100,
    "pages": 5
  },
  "links": {
    "self": "/api/v1/users?page=2&limit=20",
    "first": "/api/v1/users?page=1&limit=20",
    "prev": "/api/v1/users?page=1&limit=20",
    "next": "/api/v1/users?page=3&limit=20",
    "last": "/api/v1/users?page=5&limit=20"
  }
}
```

### Step 6: Authentication

| Method | Use Case |
|--------|----------|
| **JWT** | Stateless, mobile/web SPAs |
| **OAuth 2.0** | Third-party integrations, delegated auth |
| **API Keys** | Service-to-service, public APIs |
| **Session-based** | Server-rendered apps |

**JWT example:**

```http
GET /api/v1/users
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Step 7: Versioning

**URL versioning (recommended):**

```
/api/v1/users
/api/v2/users
```

**Header versioning:**

```
GET /api/users
Accept: application/vnd.api+json; version=1
```

### Step 8: OpenAPI 3.0 Documentation

```yaml
openapi: 3.0.0
info:
  title: User Management API
  version: 1.0.0
  description: API for managing users
servers:
  - url: https://api.example.com/v1
paths:
  /users:
    get:
      summary: List users
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
    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserCreate'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        email:
          type: string
          format: email
        created_at:
          type: string
          format: date-time
    UserCreate:
      type: object
      required:
        - name
        - email
      properties:
        name:
          type: string
        email:
          type: string
          format: email
```

## Common Query Patterns

**Filtering:**

```
GET /api/v1/users?role=admin&status=active
```

**Sorting:**

```
GET /api/v1/users?sort=-created_at,name
```

**Field selection:**

```
GET /api/v1/users?fields=id,name,email
```

**Batch operations:**

```
POST /api/v1/users/batch
{
  "operations": [
    {"action": "create", "data": {}},
    {"action": "update", "id": 123, "data": {}}
  ]
}
```

## GraphQL Alternative

If REST doesn't fit, consider GraphQL:

```graphql
type User {
  id: ID!
  name: String!
  email: String!
  posts: [Post!]!
  createdAt: DateTime!
}

type Query {
  users(page: Int, limit: Int): [User!]!
  user(id: ID!): User
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): Boolean!
}
```

## Constraints

### Required (MUST)
- **Consistency**: Use consistent naming, structure, and patterns across all endpoints
- **Versioning**: Always version your APIs from the start (URL-based recommended)
- **Security**: Implement authentication and authorization on every endpoint
- **Validation**: Validate all inputs server-side
- **Error envelopes**: Use a consistent error response format with codes and details

### Prohibited (MUST NOT)
- Verbs in URL paths (`/getUsers` → `/users`)
- Returning raw database errors to clients
- Breaking changes without a version bump
- Exposing internal IDs or implementation details in responses

## Best Practices

- **Rate limiting**: Protect against abuse with rate limits per client
- **Caching**: Use `ETag` and `Cache-Control` headers for cacheable responses
- **CORS**: Configure properly for web clients — don't use `*` in production
- **Documentation**: Keep OpenAPI specs in sync with code (generate from code or use contract testing)
- **Testing**: Test all endpoints — status codes, schemas, auth, edge cases
- **Monitoring**: Log requests, track latency, and alert on error rate spikes

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Verbs in URLs (`/getUsers`) | Use HTTP methods + noun resources |
| Inconsistent error formats | Standard error envelope with code, message, details |
| No pagination on list endpoints | Always paginate; default to sane limits (20-50) |
| Returning stack traces in production | Log server-side, return generic 500 |
| Breaking changes without version bump | Bump version or use backward-compatible additive changes |
| Exposing internal IDs (database sequence) | Use UUIDs or opaque identifiers in URLs |

## Related Skills

- **security-best-practices**: Authentication, rate limiting, CORS hardening
- **data-analysis**: API response data exploration patterns

## References

- [OpenAPI Specification](https://spec.openapis.org/oas/v3.0.3)
- [REST API Tutorial](https://restfulapi.net/)
- [GraphQL Best Practices](https://graphql.org/learn/best-practices/)
- [HTTP Status Codes (MDN)](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)
