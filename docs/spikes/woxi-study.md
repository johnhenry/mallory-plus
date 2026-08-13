# Woxi study — what a Wolfram Language reimplementation teaches the Mallory family

**Studied 2026-08-13.** Woxi ([github.com/ad-si/Woxi](https://github.com/ad-si/Woxi),
[woxi.ad-si.com](https://woxi.ad-si.com/)) is a Wolfram Language / Mathematica
reimplementation in Rust (AGPL-3.0, ~800 stars, 8k+ commits, largely
AI-agent-written). It is a similar-but-distinct neighbor: where Mallory built
its own JS-native API with adapters to incumbents (PLAN.md non-goal 11), Woxi
chose full API compatibility with one incumbent. Sources: the project site and
repo, plus two Hacker News threads
([#47155526](https://news.ycombinator.com/item?id=47155526) — the richer one —
and [#49270040](https://news.ycombinator.com/item?id=49270040)); the
r/ProgrammingLanguages thread was inaccessible to fetch tooling. Same spirit as
`cellgraph-study.md`: study the neighbor, extract what's real, file issues for
what's actionable, record what was considered and NOT adopted.

## What Woxi actually is

- Rust interpreter for a Wolfram Language subset, aimed at CLI scripting and
  notebooks. CLI (`woxi eval` / `run` / `repl`), Jupyter kernel, JupyterLite,
  browser playground (WASM build), a native notebook editor ("Woxi Studio"),
  Python bindings.
- **Testing discipline is the headline**: "All tests must pass with Woxi and
  WolframScript" — every CLI test is differential against the real Wolfram
  implementation. By the second HN post: ~26,000 unit tests + ~900 `.wls`
  snapshot tests, claiming coverage of "almost everything up to
  [Mathematica] 6.0."
- A `functions.csv` manifest tracks the full Wolfram function surface with
  per-function implementation status — a machine-readable parity ledger.
- Sells startup speed as a feature ("faster than wolframscript — no kernel
  start, no license check").

## Lessons that VALIDATE current Mallory practice (keep doing these)

1. **Differential oracles as the primary correctness strategy.** Woxi's whole
   credibility rests on the wolframscript differential suite; the HN advice
   (samwillis, foobarqux) generalizes it: with a black-box reference plus
   auto-generated inputs you get a "closed-loop" pass/fail without hand-writing
   expectations. mallory-plus already lives this: NumPy oracle (tensor-core),
   pyarrow/pandas (frame-*), scipy.signal (signal), DualNumber (autograd),
   mallory-math's own scalar functions (fft, erf). **Gap found: mallory-math's
   `Symbolic` itself has no external oracle** — see "Actionable" below.
2. **The reference isn't gospel.** Woxi's author found cases "where Mathematica
   itself didn't quite do things correctly"; HN noted even Wolfram
   users bolt on Rubi because built-in `Integrate` has gaps. Mallory's
   equivalents: the apache-arrow `Table.concat` zero-row bug (#31, filed and
   fixed here), hyparquet quirks, and the deliberate decision NOT to chase
   scipy's arbitrary SOS section-grouping byte-for-byte (signal #44) — test the
   invariant property, not the reference's incidental choices.
3. **Test counts are not evidence.** The sharpest HN pushback (hobs,
   i_cannot_hack) was at the "~5000 tests, largely AI-written" claim:
   unreviewed test count sounds good "until you think about what 5000
   unreviewed tests actually do," and "reviewing correctness is harder than
   writing correct code." The defense that actually held up was the oracle:
   a test whose expectation comes from wolframscript is self-reviewing in a
   way a hand-typed expectation is not. This is precisely why Mallory's
   convention is oracle-anchored tests over expectation dumps — keep it.
4. **Disclose coverage boundaries loudly.** A single visible gap in a flagship
   feature (`// ∫ tan(x) dx — not implemented`) read to HN as "everything
   looks very naive" (anematode). Mallory's "documented v1 simplification /
   real, disclosed gap" habit (rfft's full-spectrum note, butter's
   lowpass-only note, checkpoint's not-real-npz note) is the right defense.
5. **Positioning.** HN consensus: Mathematica's moat is "polish and overall
   experience created by consistent work over decades," not the language —
   and Octave never displaced MATLAB despite compatibility. PLAN.md's framing
   (a JS-native computation substrate, adapters-not-clones, non-goal 11)
   already avoids the "slightly different clone" trap from both directions.

## The big architectural idea (recorded, adapted, filed as someday)

The most substantive criticism in the first thread (Grosvenor): Woxi implements
polynomials/calculus **in Rust**, not in the language itself — "this will kill
you long term." The advice: keep a tiny interpreter core and express the math
as term-rewriting rules IN the hosted language, so (a) every interpreter
improvement lifts all math at once, and (b) domain experts (mathematicians)
can contribute rules without knowing the host language. porcoda confirmed the
real Mathematica core is essentially "parametrized PROLOG rules with a large
library." The counterpoint (nextaccountic, sfpotter): rules-all-the-way-down
is slow without a JIT; hand-written host-language kernels are "pre-jitted."

**Mallory mapping:** mallory-math's `Symbolic` is the Woxi shape today — its
simplify/differentiate rules are TS code over the `Expr` tagged union. The
adapted middle path (NOT rules-all-the-way-down): since `Expr` is already a
plain, serializable discriminated union, a **data-driven rewrite-rule table**
(pattern `Expr` → template `Expr`, with a small matcher) could make the
simplify/derivative rule sets declarative, individually testable, and
contributable without touching engine code — while keeping evaluation and hot
paths in TS. Filed as a "someday" design issue on `johnhenry/mallory`, not
near-term work: the current engine passes its tests and no contributor
bottleneck exists yet at this project's scale.

## Considered and NOT adopted

- **`functions.csv`-style parity manifest.** Powers Woxi's compatibility
  story, and gave them a free test corpus (an HN user donated 1,275 real
  Mathematica notebooks to run). But it only pays off when you CLAIM parity
  with an incumbent's surface. Mallory deliberately implements curated subsets
  with its own API — the honest equivalents are the per-package "v1
  scope/deferred" doc-comment sections and the issue tracker, which already
  exist. A NumPy/SciPy parity ledger would misrepresent Mallory's goals.
- **Full API compatibility with an incumbent.** Woxi's bet buys them free
  documentation, a free corpus, and users' existing code — at the price of
  inheriting every wart and a forever-chase of a moving target (they're "at
  Mathematica 6.0," which shipped in 2007). Mallory's adapter strategy stays.
- **AGPL / clean-room concerns.** Interesting thread material (Lotus v.
  Borland cited for "API clones don't infringe"), irrelevant here — Mallory is
  MIT and its API is its own.

## Code-level findings (repo scan, 2026-08-13)

Second pass: cloned the repo (shallow) and surveyed the source directly —
~500k lines under `src/` (plus ~294k of tests), 316 Rust files. **License
wall, repeated for emphasis: Woxi is AGPL-3.0 and Mallory is MIT — nothing
below may be copied as code; these are architecture observations and
methodology, which are not copyrightable.** The corpus question this pass
also settled (Rubi, not WL notebooks) is tracked as `johnhenry/mallory#16`.

### Test methodology (the best material in the repo)

1. **One harness, oracle swapped by env var.** The same test artifacts run
   against either implementation: `WOXI_USE_WOLFRAM=true` redirects the
   snapshot harness to `wolframscript -file`, the scrut CLI tests to a `wo`
   shim that dispatches to `wolframscript -c`, and even the ~26k Rust unit
   tests get oracle-verified — by a 1,900-line script that *parses the Rust
   test source* to extract `assert_eq!(interpret("…"), "…")` pairs and
   replays them through wolframscript. A conformance suite derived from the
   unit tests, with no parallel corpus to maintain. Mallory analogue: our
   oracles generate expectations rather than replaying them, which is the
   same closed loop — but the "extract-and-replay your own unit tests
   against the oracle" trick is worth remembering if a second reference
   implementation of anything ever exists.
2. **The oracle never runs in CI.** Conformance (`make test-conformance`) is
   a local, developer-run gate; CI runs only self-contained tiers. Exactly
   our skip-don't-fail convention, independently converged on.
3. **Docs are tests.** Their published documentation pages ARE the scrut
   test files (markdown with fenced `scrut` blocks + MkDocs frontmatter) —
   every example in the docs is verified against both implementations on
   every run. Genuinely attractive pattern for mallory-math's COOKBOOK.md
   someday: examples that can't rot.
4. **Differential fuzzer design** (`src/bin/diff_fuzz.rs`, ~1,450 lines) —
   the mechanics now folded into `johnhenry/mallory#14`:
   - **Spec-table generation, not grammar generation**: a curated table of
     `(function, argument-shape list)` with an `Arg` enum as the type system
     (`Num`, `IntIn(lo,hi)`, `Poly`, `PredFn`, …), ranges chosen so
     generated programs terminate and emit no messages (`Power` exponents
     bounded to ±3). Coupled args (`Part` index vs list length) are
     special-cased generators. Depth-budgeted recursion.
   - The generator **partitions its spec table against the implementation
     manifest** (functions.csv ✅ rows) and warns on drift — and a unit test
     guards the reverse direction.
   - **Batching + sentinel markers + bisection**: ~20 cases per oracle
     invocation (amortizes wolframscript's slow start), `Print` sentinel
     markers between cases, and a missing end-sentinel triggers bisecting
     the batch to distinguish a genuine hang from a cold-start flake. Any
     batch-level divergence is re-confirmed individually so batch
     scaffolding can never produce a false positive.
   - **Greedy shrinking** with strictly-size-decreasing candidates (ints →
     0/1/n±half, lists drop elements, calls hoist each argument over the
     call) and a `debug_assert` that every candidate is smaller — guaranteed
     termination, bounded by an oracle-call budget.
   - Deterministic hand-rolled RNG (SplitMix64), master + per-case seeds
     printed for exact replay; a `--oracle woxi` self-check mode that must
     report zero divergences (validates the harness itself).
   - **Exception discipline**: exact-string skip lists with a written
     justification per entry (one records that *Woxi is more accurate than
     Mathematica* on a specific `NSolve`), plus a numeric-tolerance
     escape hatch for ephemeris values — never blanket category skips.
5. **Corpus-driven burn-down.** Their changelog batches fixes per
   real-world artifact ("Fixes driven by a Wolfram Demonstration that…"),
   and `scripts-todo.md` is an activation ledger: 832 scripts, 599 verified
   byte-for-byte against the oracle, the rest triaged into buckets with
   per-bucket actions. HN bug reports from their announcement thread were
   fixed within the release cycle.
6. **Anti-pattern observed**: `functions.csv` (6,298 rows) has no schema
   validation and its `effect_level` column has visibly rotted — 15+
   inconsistent value spellings and column-shift bugs; the `rank` column is
   read by nothing. Manifests need validating tests or they rot. (Their
   fuzzer survives this by filtering only on the ✅ status column and
   keeping purity in its own curated table.)
7. Two ops-level details worth stealing: nextest per-test timeout overrides
   by name filter (an allocation-bound test gets `threads-required =
   'num-cpus'` to run in isolation instead of flaking under contention —
   the exact benchmark-contention problem we hit with the SIMD test); and a
   documented oracle warm-up step (`wolframscript` CalendarData lazy init
   paid once, outside any timed test).

### Interpreter architecture (what to do, and what not to do, in a CAS)

1. **Syntax-preserving AST = permanent tax.** Woxi's `Expr` has ~30
   variants, many encoding surface syntax (`BinaryOp`, `Map`, `Postfix`,
   `Rule`, …) alongside `FunctionCall` — so `Plus` exists in three forms
   and every consumer needs canonicalisation shims. There is no `PartialEq`;
   61 call sites compare expressions by *rendering them to strings*.
   mallory-math's `Expr` is far smaller but shares the mild form of this
   (binary `add`/`mul` nodes rather than canonical n-ary), worth
   remembering if #15's rule table ever lands: desugar in the parser, keep
   ONE application node.
2. **The "polynomials in Rust" criticism is empirically confirmed, in
   full.** Zero rewrite-rule data structures anywhere: `D` is a Rust match
   on head strings with the chain/product rules as control flow; `Simplify`
   is a generate-and-score search (candidates from Expand/Factor/Together,
   scored by leaf count) — not a rule system; `Integrate` is an *ordered
   cascade of hardcoded heuristics whose order matters*. ~53k lines of
   algebra as opaque compiled code that nothing can inspect, extend, or
   override from the language. This is the strongest possible evidence for
   `johnhenry/mallory#15`'s middle path (rules as data, algorithms as
   code) — and Woxi even shows where the boundary belongs: their
   generate-and-score Simplify and Zassenhaus factoring are genuinely
   algorithmic (fine as code); their elementary-derivative table is
   screaming to be data.
3. **User definitions stored as decomposed positional tuples** (a 6-tuple
   of parallel `Vec`s per definition) instead of first-class
   `{lhs, rhs, condition}` rules — the survey traced most of their pattern
   limitations to this. Recommendation recorded on #15: rules as
   first-class values, indexed by head.
4. **Pattern matching is real but factorial**: Flat/Orderless matching
   enumerates permutations/set-partitions outright, no discrimination nets
   or indexing. Fine at small arg counts; a known cliff.
5. **Dispatch at 6,000 builtins**: a linear chain of 22 modules, each a
   multi-thousand-line `match name` — plus a separate 1,700-line arity
   table, plus functions.csv, plus the impl: **four places to touch per
   function**, kept in sync only by agent discipline. The obvious fix they
   never made: one registry record per function (name, arity, attributes,
   impl together). mallory-math's namespace-object convention already is
   that; keep it.
6. **Numerics choices worth knowing**: rationals have no variant (they're
   `FunctionCall("Rational", …)`, string-checked at ~200 sites — ouch);
   arbitrary-precision floats are stored as *decimal strings* and re-parsed
   through astro-float at every operation (a serialization tax); `libm`
   (pure-Rust transcendentals) was adopted because **platform libm ULP
   differences broke cross-platform snapshot tests** — directly relevant to
   us, since JS engines' `Math.sin`/`Math.exp` have the same
   engine-dependence and our differential tolerances (not exact snapshots)
   are the right defense; release builds keep `overflow-checks = true`
   ("for a CAS, numeric correctness matters more than the small runtime
   cost") — mallory-plus's Rust kernels currently follow Rust's default
   (checks off in release); for f32 kernels overflow isn't the live risk,
   but worth remembering if integer kernels ever land.
7. **One genuinely portable data-structure idea**: their `ExprList` starts
   as a plain `Vec` and upgrades to a persistent RRB vector (imbl) on first
   `push_front`, with a lazily-materialized contiguous cache — turned an
   O(N²) `Prepend` chain into O(N log N) without taxing the common case.
8. **Compile-time postmortem** (`improve-compile-time.md`): five diagnosed
   causes at 500k lines, all peripheral fixes made — and their own
   conclusion names the one structural fix they never did: "split the crate
   into workspace sub-crates so edits recompile a slice." Mallory-plus's
   many-small-packages layout is already the TS equivalent; if the Rust
   crate ever grows past kernels, split early.

### WASM packaging (contrast with our seam)

Woxi ships wasm-bindgen + wasm-pack with a hand-written npm wrapper
(`woxi-wasm`): JSON-marshalled results, a host-provided `__woxi_fetch_url`
extern for `Import[url]`, and **panic recovery by re-importing the module
with a cache-busted URL** (a trapped Rust panic permanently corrupts wasm
globals; a fresh instantiation is the only cure — their playground's worker
does exactly this). mallory-plus's flat extern-C ABI (no wasm-bindgen) is
the opposite trade, chosen for zero-marshalling hot paths — both are
correct for their use ("rich API surface" vs "hot kernels"). The trapped-
instance failure mode turned out to already apply to tensor-wasm (its
`alloc`/`dealloc` `.expect()` and `solve_f32`'s internal `Vec`s are real
panic paths) — **adapted and shipped as trap poisoning** (issue #46): a
trap permanently poisons the `Kernels` instance and every later call fails
loudly instead of silently computing on corrupt memory; recovery is a
fresh `Kernels.load()` (unlike Woxi's transparent re-instantiation, our
resident `WasmTensor`s can't survive a reload, so honesty beats
transparency here).

## Actionable items (filed)

1. **SymPy differential oracle for mallory-math's `Symbolic`** — the one
   genuine testing gap this study surfaced. differentiate/simplify/
   integrate/solve currently have only hand-written expectations; a
   `sympy_oracle.py` subprocess (same pattern as mallory-plus's
   `numpy_oracle.py`/`scipy_oracle.py`, skip-don't-fail) plus a
   property-based leg (generate random `Expr` trees, compare evaluation and
   derivatives) would close the loop. → `johnhenry/mallory#14`; the code
   scan added the fuzzer mechanics (spec-table generation, batching +
   sentinels + bisection, size-monotone shrinking, exception discipline) as
   a design comment there.
2. **Declarative rewrite-rule table for `Symbolic`** (someday, above). →
   `johnhenry/mallory#15`; the code scan added the empirical confirmation
   (D/Simplify/Integrate as opaque code, definitions as positional tuples)
   and the rules-as-first-class-values-indexed-by-head recommendation as a
   comment there.
3. **Rubi-derived integration/differentiation corpus** — the corpus follow-
   up: NOT the 3GB WL-notebook archive (wrong artifact class — those are
   programs, only useful to a WL-compatible implementation), but Rubi's
   72,254-problem, MIT-licensed, Maxima-syntax test suite, whose cheapest
   tier (differentiate Rubi's own antiderivative, numerically compare to
   the integrand) exercises parse/differentiate/evaluate with no external
   tool at test time. → `johnhenry/mallory#16`.
4. **MCP server exposing Mallory evaluation to agents.** An HN commenter
   (alex7o) wrapped Woxi as an MCP server for agents and called it "an
   amazing experience" — and this family is unusually well-positioned for
   that idea (johnhenry/mcp-query, @johnhenry/mcp-gate already exist).
   A `mallory-mcp` package exposing Symbolic evaluation + tensor compute as
   MCP tools is a real, differentiated opportunity. → filed on
   `johnhenry/mallory-plus` (#45).
