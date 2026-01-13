# Axiom Classification for Phase 1 Reduction

**Total axioms in PXLv3.v: 24**

**CRITICAL FINDING:** All connectives (`⟹`, `⩪`, `⧟`, `⇌`, `⇎`) are **Parameters**, not Definitions. This means:
- Structural axioms are **primitive rules** for these abstract relations
- They **cannot be eliminated** without changing the theory's semantics
- Only potential reductions are from **overlap** or **bridging principles** that can be proven

---

## Category 1: S5 Modal Backbone (5 axioms)
**Status:** Cannot be eliminated (definitional foundation)

- `ax_K` — Distribution: □(p→q) → (□p → □q)
- `ax_T` — Reflexivity: □p → p
- `ax_4` — Transitivity: □p → □(□p)
- `ax_5` — Euclidean: ◇p → □(◇p)
- `ax_Nec` — Necessitation: p → □p

**Rationale:** These define S5 modal logic semantics. Cannot prove from weaker base without circularity.

---

## Category 2: Structural Connective Laws (9 axioms)
**Status:** CANNOT ELIMINATE (primitive theory axioms for abstract Parameters)

### Identity equivalence (3 axioms):
- `ax_ident_refl` — ∀x. x ⧟ x
- `ax_ident_symm` — x ⧟ y → y ⧟ x
- `ax_ident_trans` — x ⧟ y → y ⧟ z → x ⧟ z

**Finding:** `Ident` is a `Parameter`, not `Definition`. These axioms define its equivalence relation properties. **Cannot be removed.**

### Other structural laws (6 axioms):
- `ax_nonequiv_irrefl` — ∀x. ¬(x ⇎ x)
- `ax_inter_comm` — x ⇌ y ↔ y ⇌ x
- `ax_imp_intro` — (p→q) → p⟹q
- `ax_imp_elim` — p⟹q → (p→q)
- `ax_mequiv_intro` — (p↔q) → p⩪q
- `ax_mequiv_elim` — p⩪q → (p↔q)

**Finding:** `PImp`, `MEquiv`, `Inter`, `NonEquiv` are all `Parameters`. These axioms define their behavior. **Cannot be removed.**

---

## Category 3: PXL Core Principles (4 axioms)
**Status:** Domain-specific kernel — CHECK FOR OVERLAP with Category 2

- `A1_identity` — □(∀x. x ⧟ x)
- `A2_noncontradiction` — □(∀x,y. ¬(x⧟y ∧ x⇎y))
- `A4_distinct_instantiation` — □(𝕀₁⧟𝕀₁ ∧ 𝕀₂⧟𝕀₂ ∧ 𝕀₃⧟𝕀₃)
- `A7_triune_necessity` — □(coherence 𝕆)

**REDUCTION OPPORTUNITY:**
- `A1_identity` states: □(∀x. x⧟x)
- `ax_ident_refl` states: ∀x. x⧟x

**Question:** Can we **prove A1 from ax_ident_refl + ax_Nec?**
```coq
Lemma A1_from_refl : □ (forall x : Obj, x ⧟ x).
Proof.
  apply ax_Nec.  (* p → □p, where p = (∀x. x⧟x) *)
  intro x.
  apply ax_ident_refl.
Qed.
```
If this works, **eliminate A1_identity** and replace with this lemma. **Saves 1 axiom.**

---

## Category 4: Bridging Principles (6 axioms)
**Status:** High-level rules — CHECK IF DERIVABLE

- `modus_groundens` — □(x⧟y) → entails x P → entails y P
- `triune_dependency_substitution` — grounded_in φ 𝕀₁ → grounded_in ψ 𝕀₂ → φ⩪ψ → coherence 𝕆
- `privative_collapse` — ¬◇(entails 𝕆 P) → incoherent P
- `grounding_yields_entails` — grounded_in P x → entails x P
- `coherence_lifts_entailment` — coherence 𝕆 → entails x P → entails 𝕆 P
- `entails_global_implies_truth` — entails 𝕆 P → P

**Analysis:**
- These depend on `entails`, `grounded_in`, `incoherent`, `coherence` (all Parameters)
- Without definitions, these are **primitive bridging rules**
- **Check:** Does `entails_global_implies_truth` + A7_triune_necessity allow proving others?

---

## Phase 1B Immediate Targets

**ONLY 1 PROVABLE REDUCTION FOUND:**

1. **A1_identity** — Prove from `ax_ident_refl + ax_Nec`

**Test strategy:**
1. Add proof to PXL_Structural_Derivations.v
2. If successful, remove A1_identity from PXLv3.v
3. Update imports where needed
4. Rerun axiom_inventory.py → should see `axiom_count = 23`
5. Rerun axiom_gate.py → must still pass

---

## Revised Reduction Roadmap

**Realistic target: 24 → 23 axioms (not 13)**

The original estimate of reducing to ~13 axioms was based on the assumption that structural laws were **definitions**. Since they are **parameters** with axioms defining their behavior, the theory has a much larger irreducible kernel.

- **Phase 1B:** Eliminate A1_identity (proven from ax_ident_refl + ax_Nec)
  - Target: `24 → 23 axioms`
  
- **Phase 2:** Investigate if any bridging principles follow from combinations
  - Unlikely: all depend on undefined Parameters
  - Best case: 1–2 axioms saved
  - Target: `23 → 21–22 axioms`
  
- **Final kernel:** ~21 axioms = 5 (S5) + 9 (structural laws for Parameters) + 3 (PXL core) + 4–6 (irreducible bridging)

---

**Current gate status:**
- ✅ 24 axioms <= 49 budget
- ✅ pxl_excluded_middle: 0 assumptions
- ✅ trinitarian_optimization: 0 assumptions

**Next action:** Attempt A1_identity → A1_from_refl proof in PXL_Structural_Derivations.v
