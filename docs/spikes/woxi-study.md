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

## Actionable items (filed)

1. **SymPy differential oracle for mallory-math's `Symbolic`** — the one
   genuine testing gap this study surfaced. differentiate/simplify/
   integrate/solve currently have only hand-written expectations; a
   `sympy_oracle.py` subprocess (same pattern as mallory-plus's
   `numpy_oracle.py`/`scipy_oracle.py`, skip-don't-fail) plus a
   property-based leg (generate random `Expr` trees, compare evaluation and
   derivatives) would close the loop. → filed on `johnhenry/mallory`.
2. **Declarative rewrite-rule table for `Symbolic`** (someday, above). →
   filed on `johnhenry/mallory`.
3. **MCP server exposing Mallory evaluation to agents.** An HN commenter
   (alex7o) wrapped Woxi as an MCP server for agents and called it "an
   amazing experience" — and this family is unusually well-positioned for
   that idea (johnhenry/mcp-query, @johnhenry/mcp-gate already exist).
   A `mallory-mcp` package exposing Symbolic evaluation + tensor compute as
   MCP tools is a real, differentiated opportunity. → filed on
   `johnhenry/mallory-plus`.
