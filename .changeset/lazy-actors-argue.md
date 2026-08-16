---
"mallory-tensor-autograd": minor
---

Fixes johnhenry/mallory-plus#89: `optim.SGD` gains an optional `momentum`/`nesterov` option, following PyTorch's own update convention (`buf = momentum*buf + grad`, Nesterov's lookahead `d_p = grad + momentum*buf` applied after the buffer update). Both default to off (`0`/`false`), so `new SGD(params, { lr })` is byte-identical to the pre-#89 plain-SGD update -- no existing caller's behavior changes. Constructing with `nesterov: true` and no (or zero) `momentum` throws a `RangeError`, since Nesterov's lookahead is meaningless without a momentum term to look ahead with.
