# NLM V0.16.1 — Terminal Program Closure Audit

## Status

`V0161_PROGRAM_EXECUTION_CYCLE_CLOSED`

This document closes the **current V0.16.1 research lineage**. It does not claim that the integrated Nolane Living Model architecture has been validated. The closure is a falsification-program closure: every V0.16.1 experiment now has an evidence-backed result or an explicit dependency disposition, and the current lineage has no authorized path to EXP-300 after EXP-299 failed its frozen scaffold-removal gate.

Authoritative repository base before this closure: `main@bd8696a3263f1be2c624a3ee995586a8acc455a0`.

Frozen Stage-A protocol digest: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.

Reviewer-hardened V0.16.1 source copy audited for this closure:

- filename: `NOLANE-LIVING-MODEL-V0.16.1-100M-ADVERSARIAL-EPISTEMIC-CLOSURE-REVIEWER-HARDENED(20260913-121002).md`
- SHA-256: `55ed896bee0306a5729edc5c52eb58fbe01c12e367ced14f2096a52dab85e5f7`
- rendered line count: `32,575`

## What "complete" means here

V0.16.1 explicitly requires negative results to simplify the architecture rather than being treated as implementation failures. It also requires oracle/headroom gates before learned downstream complexity and blocks scaling unless the prerequisite gates survive.

Therefore V0.16.1 completion is **not** defined as mechanically executing all EXP-277..EXP-300 IDs. It is defined as:

1. freeze the Stage-A research contract before confirmatory evidence;
2. execute the decisive gates without post-result retuning;
3. follow successor authority only when the frozen parent result grants it;
4. stop downstream work when a parent kill/negative disposition removes its justification;
5. preserve positive scoped results without laundering them into integrated success;
6. record all blocked/dependency-review experiments explicitly;
7. close the current lineage when no authorized successor remains.

Under that definition, the current V0.16.1 cycle is terminally closed.

## Decisive Stage-A gates

The six experiments named by the reviewer-hardened spec as the next scientific action have all received a disposition:

| Experiment | Frozen role | Final current-lineage disposition |
|---|---|---|
| EXP-277 | oracle constraint/factor headroom | `SCIENTIFICALLY_CLOSED_KILL` |
| EXP-279 | propagation vs branch vs hybrid | `DEVELOPMENT_CLOSED_NEGATIVE` |
| EXP-282 | explicit belief vs recurrent hidden state | `SCIENTIFICALLY_CLOSED_KILL` |
| EXP-286 | oracle conflict-core value | `SCIENTIFICALLY_CLOSED_PROMOTE` |
| EXP-289 | episode-local nogood learning | `SCIENTIFICALLY_CLOSED_PROMOTE` |
| EXP-297 | encoding fidelity court | `SCIENTIFICALLY_CLOSED_PROMOTE` |

The Stage-A execution-contract debt in the source specification is therefore no longer open for these six gates.

## Authorized downstream seams and their closures

Only downstream questions that actually received parent authority were opened.

### Conflict lineage

`EXP-286 -> EXP-287`

EXP-286 established scoped oracle conflict-core value. EXP-287 then tested learned conflict localization under its own frozen DEVELOPMENT court and closed:

`LOCALIZATION_VALUE_NOT_ESTABLISHED`

Consequence: EXP-288 remains blocked. No threshold, top-k, seed, sample-size or same-hypothesis rerun is authorized.

### Nogood / structural-transfer lineage

`EXP-289 -> EXP-290`

EXP-289 established scoped value for episode-local nogoods. EXP-290 then tested learned structural clause transfer and closed:

`STRUCTURAL_CLAUSE_TRANSFER_NOT_ESTABLISHED`

Consequence: EXP-291 has no authority on this lineage; EXP-292..EXP-296 remain blocked behind the missing trustworthy counterexample/lemma-transfer substrate.

### Fidelity / native-internalization lineage

`EXP-297 -> EXP-298 -> EXP-299`

EXP-297 established bounded Encoding Fidelity Court value. EXP-298 then closed recurrent positive for controlled-domain fidelity transfer:

`CROSS_DOMAIN_FIDELITY_TRANSFER_RECURRENT`

That result authorized only EXP-299 design/preregistration. EXP-299 executed exactly once under its frozen court and closed:

`NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT`

with:

- `established_root_count=0/4`
- `successor_design_authorized=false`
- `authorization_scope=NONE`
- `exp300_execution_authorized=false`

Consequence: EXP-300 is blocked on the current lineage.

## Full EXP-277..EXP-300 terminal authority map

All 24 experiment IDs now have an explicit program disposition.

| ID | Status | Terminal meaning for V0.16.1 current lineage |
|---|---|---|
| EXP-277 | `SCIENTIFICALLY_CLOSED_KILL` | frozen oracle-headroom hypothesis killed |
| EXP-278 | `DEPENDENCY_REVIEW_REQUIRED` | old compiler justification removed by EXP-277; no execution authority |
| EXP-279 | `DEVELOPMENT_CLOSED_NEGATIVE` | current routing/representation line closed |
| EXP-280 | `DEPENDENCY_REVIEW_REQUIRED` | no inherited authority from EXP-279 |
| EXP-281 | `DEPENDENCY_REVIEW_REQUIRED` | no viable inherited propagation/equilibration substrate |
| EXP-282 | `SCIENTIFICALLY_CLOSED_KILL` | frozen explicit-belief comparison killed |
| EXP-283 | `DEPENDENCY_REVIEW_REQUIRED` | requires a scientifically distinct post-EXP-282 hypothesis |
| EXP-284 | `DEPENDENCY_REVIEW_REQUIRED` | requires an independent preregistered hypothesis |
| EXP-285 | `DEPENDENCY_REVIEW_REQUIRED` | requires a standalone preregistered residual-typing hypothesis |
| EXP-286 | `SCIENTIFICALLY_CLOSED_PROMOTE` | scoped oracle conflict-core value established |
| EXP-287 | `DEVELOPMENT_CLOSED_NEGATIVE` | learned localization not established |
| EXP-288 | `BLOCKED_ON_PARENT` | EXP-287 did not authorize successor |
| EXP-289 | `SCIENTIFICALLY_CLOSED_PROMOTE` | scoped episode-local nogood value established |
| EXP-290 | `DEVELOPMENT_CLOSED_NEGATIVE` | learned structural transfer not established |
| EXP-291 | `BLOCKED_ON_PARENT` | EXP-290 did not authorize successor |
| EXP-292 | `BLOCKED_ON_PARENT` | missing trustworthy counterexample/lemma predecessor |
| EXP-293 | `BLOCKED_ON_PARENT` | missing viable lemma generator/library |
| EXP-294 | `BLOCKED_ON_PARENT` | missing viable lemma eligibility/generation chain |
| EXP-295 | `BLOCKED_ON_PARENT` | no authorized live lemma economy to garbage-collect |
| EXP-296 | `BLOCKED_ON_PARENT` | no authorized reusable lemma dependency substrate |
| EXP-297 | `SCIENTIFICALLY_CLOSED_PROMOTE` | bounded fidelity-court value established |
| EXP-298 | `DEVELOPMENT_CLOSED_POSITIVE` | controlled-domain recurrent transfer established; scope remains bounded |
| EXP-299 | `DEVELOPMENT_CLOSED_NEGATIVE` | native scaffold-free advantage not established |
| EXP-300 | `BLOCKED_ON_PARENT` | current lineage lacks required native gains and successor authority |

Counts:

- direct scientific/development courts with final evidence disposition: `10/24`;
- dependency-review or blocked experiments correctly withheld from execution: `14/24`;
- experiment IDs lacking an authority disposition: `0/24`.

The 14 withheld experiments are **not unfinished executions**. Under the frozen dependency logic, running them without a new preregistered hypothesis would violate the spec.

## V0.16 scale-gate outcome

The V0.16 scale gate required, among other things, material oracle constraint headroom, viable learned downstream structure, safe learned conflict localization/backjump, positive lemma-economy evidence, cross-domain transfer, positive lifecycle VCPF and model-native gains after scaffold removal.

The current lineage does not satisfy that conjunction:

- EXP-277 killed the frozen oracle-headroom claim;
- EXP-279 failed to establish the current routing/representation line;
- EXP-287 failed learned conflict localization;
- EXP-290 failed learned structural clause transfer;
- lemma-economy EXP-291..296 therefore never received authority;
- EXP-298 was positive only in a bounded controlled-domain DEVELOPMENT scope;
- EXP-299 failed the native scaffold-removal requirement;
- EXP-300 therefore cannot validly run as the current-lineage integrated comparison.

Canonical scale disposition:

`DO_NOT_SCALE_CURRENT_V0161_LINEAGE`

This is not a claim that every future NLM architecture is impossible. It is a decision that **this frozen lineage has not earned scaling or an integrated EXP-300 court**.

## Resolution of the V0.16.1 irreducible-debt list

The reviewer-hardened source listed ten empirical debts. They now classify as follows:

1. **Oracle constraint headroom:** measured for the frozen Stage-A question and closed negative by EXP-277.
2. **Learned constraint compilation preserving fidelity:** original straight-line justification lost after EXP-277; remains a future distinct-hypothesis debt, not an authorized V0.16.1 run.
3. **Explicit belief vs strong recurrent state:** measured and closed negative by EXP-282.
4. **Reliable learned conflict localization:** measured in EXP-287 V1 and not established.
5. **Structural learned-clause transfer:** measured in EXP-290 V1 and not established.
6. **Long-lived lemma economy net utility:** not authorized after EXP-290; remains future-research debt outside the current lineage.
7. **Broad open-domain semantic formalization:** still unverified; EXP-297/298 support only bounded scopes.
8. **Integrated architecture vs simpler monolith:** not authorized because EXP-299 failed; EXP-300 remains blocked.
9. **Which older learning components survive neural ablation:** remains broader future NLM research debt; historical V0.1..V0.14 proposals are not silently promoted by V0.16.1.
10. **Scaling beyond the 100M wind tunnel:** remains untested and unauthorized for the current lineage.

A debt does not have to be answered positively for the research cycle to close. It must be either measured, killed, bounded, blocked by a failed prerequisite, or explicitly carried forward as new research. That condition is now met.

## W5 boundary

W5 remains a process/research-pressure gate, not a scientific oracle. Its historical open status is preserved rather than cosmetically forced to pass.

No fake active residency, independent challenger, robustness world, verification identity, or quality score is invented for terminal closure.

The V0.16.1 program is closed by frozen empirical authority and dependency logic, **not** by claiming a W5 pass.

## What is established

The repository now supports these narrow conclusions:

- the Stage-A protocol and evidence machinery are executable and frozen;
- several scoped oracle/mechanism hypotheses were genuinely tested;
- EXP-286, EXP-289 and EXP-297 achieved scoped scientific promotion dispositions;
- EXP-298 achieved a bounded controlled-domain DEVELOPMENT-positive recurrent disposition;
- EXP-277 and EXP-282 were scientifically killed in their frozen scopes;
- EXP-279, EXP-287, EXP-290 and EXP-299 closed negative in their frozen DEVELOPMENT scopes;
- EXP-299's negative result removes current-lineage authority for EXP-300;
- the current architecture is therefore required to shrink rather than be force-scaled.

## What is not established

This closure does **not** establish:

- integrated NLM superiority;
- a successful full 100M living model;
- broad open-domain semantic authority;
- reliable learned conflict localization;
- a positive lifelong lemma economy;
- scaffold-free model-native fidelity advantage;
- superior lifecycle VCPF;
- general continual-learning/plasticity success across years;
- scaling benefit at 1B/100B/frontier size;
- consciousness or subjective life.

## Completion accounting

Two different completion axes must not be conflated.

### Program/spec execution closure

`24/24` V0.16 experiment IDs have an explicit authority disposition and every authorized current-lineage successor has been either executed or blocked by its frozen parent outcome.

**V0.16.1 current-lineage program closure: 100%.**

### Architecture success / empirical maturity

There is deliberately no synthetic "100% success" score. The integrated architecture did not earn EXP-300 authority and remains unverified as a whole.

A negative or blocked result is still a completed research outcome; it is not evidence of capability success.

## Final authority rule

Within V0.16.1, there is no remaining authorized experiment execution after this closure.

Future work must begin under a **new, scientifically distinct hypothesis/version** if it wishes to revisit any of these areas:

- constraint representation/compiler design after EXP-277;
- routing/propagation representation after EXP-279;
- belief/recovery mechanisms after EXP-282;
- conflict localization after EXP-287;
- structural transfer / lemma economy after EXP-290;
- model-native scaffold-free fidelity after EXP-299;
- integrated 100M comparison after a future viable native candidate exists.

Such work is not a continuation/rerun of the failed frozen V1 court. It must receive a new preregistration, new pre-data gate and its own authority chain.

## Terminal disposition

```text
V0.16.1 specification/process contract: CLOSED
V0.16.1 current experiment dependency graph: CLOSED
V0.16.1 current-lineage scaling authority: DENIED
EXP-300 current-lineage execution authority: DENIED
Integrated NLM capability claim: UNVERIFIED
Future scientifically distinct NLM research: ALLOWED UNDER A NEW VERSION/HYPOTHESIS
```

This is the intended endpoint of a falsification-first specification: the program can finish cleanly even when the architecture it tested does not earn promotion.