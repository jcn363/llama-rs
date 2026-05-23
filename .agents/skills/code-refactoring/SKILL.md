---
name: code-refactoring
description: "Use when discovering complex or duplicated code during code review, cleaning up code before adding new features, removing root causes after bug fixes, or resolving technical debt - covers Extract Method, DRY removal, conditional-to-polymorphism conversion, parameter object introduction, SOLID principles, behavior validation workflow, and multi-agent verification."
---

# Code Refactoring

## Overview

Refactor code while preserving behavior, improving clarity, and reducing complexity — using Extract Method, DRY, polymorphism, parameter objects, and SOLID principles with test-first validation.

## When to Use

- **Code review**: Discovering complex or duplicated code
- **Before adding new features**: Cleaning up existing code
- **After bug fixes**: Removing root causes
- **Resolving technical debt**: Regular refactoring

## Core Patterns

### Step 1: Extract Method

Break long functions into smaller named units.

**Before (long function):**

```typescript
function processOrder(order: Order) {
  // Validation
  if (!order.items || order.items.length === 0) {
    throw new Error('Order must have items');
  }
  if (!order.customerId) {
    throw new Error('Order must have customer');
  }

  // Price calculation
  let total = 0;
  for (const item of order.items) {
    total += item.price * item.quantity;
  }
  const tax = total * 0.1;
  const shipping = total > 100 ? 0 : 10;
  const finalTotal = total + tax + shipping;

  // Inventory check
  for (const item of order.items) {
    const product = await db.product.findUnique({ where: { id: item.productId } });
    if (product.stock < item.quantity) {
      throw new Error(`Insufficient stock for ${product.name}`);
    }
  }

  // Create order
  const newOrder = await db.order.create({
    data: { customerId: order.customerId, items: order.items, total: finalTotal, status: 'pending' }
  });
  return newOrder;
}
```

**After (extracted methods):**

```typescript
async function processOrder(order: Order) {
  validateOrder(order);
  const total = calculateTotal(order);
  await checkInventory(order);
  return await createOrder(order, total);
}

function validateOrder(order: Order) {
  if (!order.items || order.items.length === 0) throw new Error('Order must have items');
  if (!order.customerId) throw new Error('Order must have customer');
}

function calculateTotal(order: Order): number {
  const subtotal = order.items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  const tax = subtotal * 0.1;
  const shipping = subtotal > 100 ? 0 : 10;
  return subtotal + tax + shipping;
}

async function checkInventory(order: Order) {
  for (const item of order.items) {
    const product = await db.product.findUnique({ where: { id: item.productId } });
    if (product.stock < item.quantity) throw new Error(`Insufficient stock for ${product.name}`);
  }
}

async function createOrder(order: Order, total: number) {
  return await db.order.create({
    data: { customerId: order.customerId, items: order.items, total, status: 'pending' }
  });
}
```

### Step 2: Remove Duplication (DRY)

Extract shared logic into a parameterized function.

**Before:**

```typescript
async function getActiveUsers() {
  return await db.user.findMany({
    where: { status: 'active', deletedAt: null },
    select: { id: true, name: true, email: true }
  });
}

async function getActivePremiumUsers() {
  return await db.user.findMany({
    where: { status: 'active', deletedAt: null, plan: 'premium' },
    select: { id: true, name: true, email: true }
  });
}
```

**After:**

```typescript
type UserFilter = { plan?: string };

async function getActiveUsers(filter: UserFilter = {}) {
  return await db.user.findMany({
    where: { status: 'active', deletedAt: null, ...filter },
    select: { id: true, name: true, email: true }
  });
}

const allActiveUsers = await getActiveUsers();
const premiumUsers = await getActiveUsers({ plan: 'premium' });
```

### Step 3: Replace Conditional with Polymorphism

Replace long if-else chains with a strategy pattern.

**Before:**

```typescript
class PaymentProcessor {
  process(payment: Payment) {
    if (payment.method === 'credit_card') {
      return this.chargeCreditCard(this.tokenizeCard(payment.card), payment.amount);
    } else if (payment.method === 'paypal') {
      return this.getPayPalApproval(this.createPayPalOrder(payment.amount));
    } else if (payment.method === 'bank_transfer') {
      return this.initiateBankTransfer(payment.account, payment.amount);
    }
  }
}
```

**After:**

```typescript
interface PaymentMethod {
  process(payment: Payment): Promise<PaymentResult>;
}

class CreditCardPayment implements PaymentMethod {
  async process(payment: Payment): Promise<PaymentResult> {
    const cardToken = await this.tokenizeCard(payment.card);
    return await this.chargeCreditCard(cardToken, payment.amount);
  }
}

class PayPalPayment implements PaymentMethod {
  async process(payment: Payment): Promise<PaymentResult> {
    const order = await this.createPayPalOrder(payment.amount);
    return await this.getPayPalApproval(order);
  }
}

class BankTransferPayment implements PaymentMethod {
  async process(payment: Payment): Promise<PaymentResult> {
    return await this.initiateBankTransfer(payment.account, payment.amount);
  }
}

class PaymentProcessor {
  private methods = new Map<string, PaymentMethod>([
    ['credit_card', new CreditCardPayment()],
    ['paypal', new PayPalPayment()],
    ['bank_transfer', new BankTransferPayment()]
  ]);

  async process(payment: Payment): Promise<PaymentResult> {
    const method = this.methods.get(payment.method);
    if (!method) throw new Error(`Unknown payment method: ${payment.method}`);
    return await method.process(payment);
  }
}
```

### Step 4: Introduce Parameter Object

Group long parameter lists into a typed object.

**Before:**

```typescript
function createUser(
  name: string, email: string, password: string,
  age: number, country: string, city: string,
  postalCode: string, phoneNumber: string
) { /* ... */ }
```

**After:**

```typescript
interface UserProfile { name: string; email: string; password: string; age: number; }
interface Address { country: string; city: string; postalCode: string; }
interface CreateUserParams { profile: UserProfile; address: Address; phoneNumber: string; }

function createUser(params: CreateUserParams) {
  const { profile, address, phoneNumber } = params;
  // ...
}

createUser({
  profile: { name: 'John', email: 'john@example.com', password: 'xxx', age: 30 },
  address: { country: 'US', city: 'NYC', postalCode: '10001' },
  phoneNumber: '+1234567890'
});
```

### Step 5: Apply SOLID Principles (Single Responsibility)

**Before (multiple responsibilities in one class):**

```typescript
class User {
  constructor(public name: string, public email: string) {}
  save() { /* Save to DB */ }
  sendEmail(subject: string, body: string) { /* Send email */ }
  generateReport() { /* Generate report */ }
}
```

**After (separated concerns):**

```typescript
class User {
  constructor(public name: string, public email: string) {}
}

class UserRepository {
  save(user: User) { /* Save to DB */ }
}

class EmailService {
  send(to: string, subject: string, body: string) { /* Send email */ }
}

class UserReportGenerator {
  generate(user: User) { /* Generate report */ }
}
```

## Behavior Validation Workflow

### Step A: Understand Current Behavior

Document behavior before changing anything:

```markdown
## Behavior Analysis

### Inputs
- [list of input parameters, types, and constraints]

### Outputs
- [return values, side effects]

### Invariants
- [conditions that must always be true]
- [edge cases]

### Dependencies
- [external dependencies, state dependencies]
```

### Step B: Validate After Refactoring

```bash
# 1. Run tests
npm test -- --coverage

# 2. Type check
npx tsc --noEmit

# 3. Lint check
npm run lint

# 4. Compare with previous behavior (snapshot tests)
npm test -- --updateSnapshot
```

### Step C: Document Changes

```markdown
## Refactoring Summary

### Changes Made
1. [Change 1]: [reason]
2. [Change 2]: [reason]

### Behavior Preserved
- [x] Same input → same output
- [x] Same side effects
- [x] Same error handling

### Risks & Follow-ups
- [potential risks]
- [follow-up tasks]

### Test Status
- [ ] Unit tests: passing
- [ ] Integration tests: passing
- [ ] E2E tests: passing
```

## Refactoring Checklist

- [ ] Function does one thing only (SRP)
- [ ] Function name clearly describes what it does
- [ ] Function is 20 lines or fewer (guideline)
- [ ] 3 or fewer parameters
- [ ] No duplicate code (DRY)
- [ ] if nesting is 2 levels or fewer
- [ ] No magic numbers (extract as constants)
- [ ] Understandable without comments (self-documenting)

## Multi-Agent Workflow

| Round | Agent | Role |
|-------|-------|------|
| Validation | Orchestrator | Validate behavior preservation checklist |
| Analysis | Analyst/Explorer | Complexity and duplication analysis |
| Verification | Executor/Fixer | Test or static analysis verification |

## Constraints

### Required (MUST)
- **Test first**: Write tests before refactoring
- **Small steps**: Change one thing at a time
- **Behavior preservation**: No functional changes during refactoring

### Prohibited (MUST NOT)
- **Multiple tasks simultaneously**: No refactoring + feature addition at the same time
- **Refactoring without tests**: Risk of undetected regression

## Best Practices

- **Boy Scout Rule**: Leave code cleaner than you found it
- **Refactoring timing**: Red-Green-Refactor (TDD cycle)
- **Incremental improvement**: Consistency over perfection
- **Behavior preservation**: Refactoring involves zero functional changes
- **Small commits**: Commit in focused, revertable units

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Refactoring and adding features at the same time | Separate into two distinct changes |
| No tests before refactoring | Write behavior-capturing tests first |
| Changing too much at once | One extraction or rename per commit |
| Renaming while restructuring | Do rename and structural moves separately |
| Forgetting edge cases | Document invariants pre-refactoring and verify post-refactoring |
| Premature abstraction | Wait for duplication to appear before extracting |

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| Tests fail after refactor | Behavior changed unintentionally | Revert and isolate the change, then retry |
| Code still complex | Multiple responsibilities mixed | Extract into smaller units with clear boundaries |
| Performance regression | Inefficient abstraction introduced | Profile the hot path and optimize |
| Difficult to name the function | It does too many things | Split further until naming is obvious |

## Related Skills

- **code-review**: Pre-refactoring analysis of code quality issues
- **rust-testing**: Test patterns for behavior preservation validation
- **rust-best-practices**: Idiomatic patterns for Rust refactoring

## References

- *Refactoring* (Martin Fowler)
- *Clean Code* (Robert C. Martin)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
