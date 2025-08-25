# Changelog: Implementing Specular Manifold Sampling (SMS) in Cycles

This document details changes to Cycles for Specular Manifold Sampling (SMS), building on the Manifold Next Event Estimation (MNEE) framework. The current implementation focuses on **refractive** specular paths, handles **multi-scatter chains**, and offers **biased and unbiased** estimators as per Zeltner et al. [2020].

The changes are mostly within the cycles' kernel in `intern/cycles/kernel`; the paths below are relative to this directory for simplicity.

An attempt to fully support **reflection** was made, but solver convergence issues (under 1% success for reflective scenes) led to prioritizing robust refraction. Core manifold functions have reflection capabilities, but SMS path generation currently doesn't use them.

The existing MNEE framework (`integrator/mnee.h`) provides the core manifold walk solver. New SMS logic for stochastic initialization, multi-scatter chain finding, and probability estimation is in `integrator/sms.h`. **Angle difference constraints** are partially implemented for the Newton solver, alongside the existing half-vector constraints.

## Core Manifold Function Updates (`integrator/mnee.h`)

Functions in `integrator/mnee.h` were updated to support SMS, including different constraint types and, in principle, reflection/refraction.

Key updates:

1.  **Explicit `reflection` Parameter:** Functions like `mnee_compute_hv_constraint_derivatives`, `mnee_newton_solver`, `mnee_eval_bsdf_contribution`, and `mnee_compute_transfer_matrix` now take a `reflection` boolean. This allows them to switch internal physics:
    *   If `reflection` is true and the BSDF is reflective, they use reflection-specific formulas (e.g., half-vector `H = wi + wo`).
    *   If `reflection` is false, they use refraction-specific formulas (e.g., generalized half-vector `H = -(wi + eta*wo)`).
    *   **SMS currently calls these functions with `reflection = false`, focusing on refractive interactions.**

2.  **Angle Difference Constraint Implementation:**
    *   New helper functions for angle difference calculations were added (`ad_reflect`, `ad_d_reflect`, `ad_refract`, `ad_d_refract`, `ad_sphcoords`, `ad_d_sphcoords`).
    *   `mnee_compute_ad_constraint_derivatives` was implemented to compute constraints based on angle differences (SMS paper Eq. 12), handling reflection/refraction based on BSDF type and the passed `reflection` flag. It uses a two-strategy approach for robustness (e.g., with TIR).
    *   `mnee_newton_solver` now accepts a `caustics_constraint_derivatives` parameter (UI controlled) to choose between half-vector (`mnee_compute_hv_constraint_derivatives`) or angle difference (`mnee_compute_ad_constraint_derivatives`) constraints.
    *   **SMS uses angle difference constraints for the Newton solver when selected via UI, but currently only for refractive paths as it calls the solver with `reflection = false`.**

3.  **Stochastic Vertex Initialization in `mnee_setup_manifold_vertex`:**
    *   `mnee_setup_manifold_vertex` now has an optional `rng_state` parameter. If provided, it samples random barycentric coordinates on the triangle, instead of using exact intersection barycentrics. This is used by SMS for geometric randomization.

4.  **Generalized BSDF Sampling and Evaluation:**
    *   `mnee_sample_bsdf_dh`: Extended to sample offsets for Beckmann reflection BSDFs.
    *   `mnee_eval_bsdf_contribution` and `mnee_compute_transfer_matrix`: Generalized with the `reflection` flag.
    *   **As SMS calls these with `reflection = false`, their reflective logic is currently unused by SMS. Contribution and MIS for SMS paths use the refractive, half-vector formulation.**

## SMS Implementation (`integrator/sms.h`)

The new `integrator/sms.h` contains the main SMS logic for biased and unbiased multi-scatter refractive caustics.

1.  **Constants and Structs:**
    *   Defines `SMS_MAX_TRIALS`, `SMS_BIASED_BUDGET`, `SMS_EPSILON`.
    *   `SMSUniqueSolution` struct for biased SMS, with `hash_float3` and `sms_is_duplicate_solution` helpers.

2.  **`sms_find_caster_chain` Helper Function:**
    *   Finds a sequence of potential SMS caster vertices.
    *   Casts a probe ray from receiver to light.
    *   Iterates intersections, validating casters:
        *   Must be `SD_OBJECT_CAUSTICS_CASTER`, not the receiver, within limits.
        *   Requires `PRIMITIVE_TRIANGLE` and `SHADER_SMOOTH_NORMAL`.
        *   **Crucially, it only selects closures satisfying `CLOSURE_IS_REFRACTION(sc->type) || CLOSURE_IS_GLASS(sc->type)`, thus limiting SMS to refractive/glass chains.**
        *   Stores valid intersection data.
    *   Checks path length limits. Returns `vertex_count`.

3.  **Stochastic Initialization (Biased & Unbiased Trials):**
    *   Achieved by:
        *   Random microfacet normal (`h_offset`) via `mnee_sample_bsdf_dh` for rough surfaces.
        *   Random geometric position on primitives (random barycentrics via `rng_state` in `mnee_setup_manifold_vertex`).
    *   The sequence of *caster primitives* is still found by `sms_find_caster_chain`'s deterministic probe ray.

4.  **Unbiased SMS (`integrate_sms_unbiased`):**
    *   Implements Algorithm 2 (SMS paper) for multi-scatter refractive paths.
    *   Calls `sms_find_caster_chain`.
    *   **Reference Path:** Stochastic vertex initialization for `vertices_ref`. Runs `mnee_newton_solver` (with `reflection = false`). If no convergence, returns zero. Stores `solution_p_ref`.
    *   **Bernoulli Trials:** Loops up to `SMS_MAX_TRIALS`, re-initializing `vertices_trial` and running solver. If converged and matches `solution_p_ref`, breaks. Increments `inv_prob_estimate` on failure.
    *   **Final Result:** If `inv_prob_estimate > 0.0f`, returns `mnee_path_contribution(vertices_ref, reflection = false) * inv_prob_estimate`.

5.  **Biased SMS (`integrate_sms_biased`):**
    *   Implements Algorithm 3 (SMS paper) for multi-scatter refractive paths.
    *   Calls `sms_find_caster_chain`.
    *   Fixed budget (`SMS_BIASED_BUDGET`). Stores unique solutions.
    *   **Trial Loop:** For each trial, stochastic vertex initialization for `vertices_trial`. Runs `mnee_newton_solver` (with `reflection = false`).
    *   If converged and unique (first vertex state), calculates and stores `mnee_path_contribution(vertices_trial, reflection = false)`.
    *   **Final Result:** Sums stored contributions.

## Integration Changes (`integrator/shade_surface.h` context)

Integration is in `integrate_surface_direct_light`:

1.  **Caustics Sampling Strategy Selection:**
    *   For `SD_OBJECT_CAUSTICS_RECEIVER` surfaces and `use_caustics` lights:
        *   If SMS (Biased/Unbiased) is selected, calls the corresponding `integrate_sms_*` function.
        *   If SMS returns non-zero contribution: clamps, writes via `film_write_direct_light_sms`, and **returns**, skipping NEE.
        *   If SMS fails or MNEE is selected, falls back to `kernel_path_mnee_sample`.
        *   Standard NEE proceeds if all manifold methods fail/disabled.
2.  **New Film Writing Function (`film_write_direct_light_sms`):**
    *   Added to correctly write SMS direct light to film passes (Combined, Lightgroups, Diffuse/Glossy/Transmission Direct).

## Supporting Type Changes (`svm/types.h`)

1.  **Reflection Macro:** `CLOSURE_IS_REFLECTION(type)` (Beckmann, GGX).
2.  **SMS Compatibility Macro:** `CLOSURE_IS_SMS_COMPATIBLE(type)` (Reflection, Refraction, Glass).
    *   **Note:** While this macro includes reflective types, `sms_find_caster_chain` currently filters them out.

## Current State and Limitations

SMS in Cycles enables multi-scatter refractive caustics with biased/unbiased estimators.

**What Works / Fully Supported:**

*   **Refractive Caustics:** Multi-scatter paths with refraction/glass BSDFs.
*   **Biased and Unbiased Estimators:** For these refractive chains.
*   **Half-Vector Constraints:** For refraction (MNEE's basis).
*   **Stochastic Initialization:** Random microfacet normals + random points on pre-identified primitives.
*   **Core Manifold Engine:** `integrator/mnee.h` functions can, in principle, handle reflection.

**What is Partially Supported / Not Optimal:**

1.  **Angle Difference Constraints:**
    *   **Current State:** Implemented (`mnee_compute_ad_constraint_derivatives`) and selectable via UI for the `mnee_newton_solver` on **refractive paths**. Aims to improve solver robustness.
    *   **Limitation (Solver):** The implementation is flawed. It incorrectly applies the angle-difference math to the smooth shading normal (`n`) instead of the microfacet normal (`m`), which is required for physically correct scattering from microfacet BSDFs. This makes the solver target an incorrect state, limiting its effectiveness.
    *   **Limitation (Contribution):** Path contribution (`mnee_path_contribution`) and MIS (`mnee_eval_bsdf_contribution`) still use the half-vector formulation. For full angle-difference SMS, these would need updates.

**What Doesn't Work / Not Supported Yet:**

1.  **Reflection Support in SMS:**
    *   **Background:** Core `mnee.h` functions were extended for reflection, but this revealed fundamental issues with the solver. The low success rate is caused by numerical instability in the Half-Vector Constraint (`H = wi + wo`). In common reflective configurations, `wi` and `wo` are nearly anti-parallel, causing their sum `H` to approach a zero vector. The required normalization `1/len(H)` then explodes, filling the solver's Jacobian matrix with `inf` or `NaN` values and causing immediate divergence.
    *   **Current State:** Disabled in SMS due to this instability. `sms_find_caster_chain` only selects refractive/glass BSDFs, and SMS integration functions call core solvers with `reflection = false`.
    *   **Consequence:** SMS **does not currently render reflective caustics**.

2.  **Stochastic Initialization Strategy (Global Geometry):**
    *   **Current State:** `sms_find_caster_chain` uses a deterministic probe ray to find caster primitives. Randomness is on these primitives.
    *   **Limitation:** The SMS paper describes more global geometric seed path sampling (e.g., randomly picking any caster object). Not yet implemented.

3.  **Two-Stage Manifold Walks (Normal Maps):** Not implemented (SMS paper Section 4.5).

4.  **Glint Rendering Specialization:** Not implemented (SMS paper Section 4.6).

5.  **MIS Integration with Other Techniques / Advanced MIS:**
    *   **Current State:** SMS replaces NEE if successful. Fallback otherwise. Internal MIS terms in `mnee_eval_bsdf_contribution` are from half-vector MNEE.
    *   **Limitation:** No MIS combination of SMS with NEE/BSDF sampling. MIS calculations not updated for full SMS (e.g., angle difference or potential reflection).

## Summary

Cycles now has multi-scatter Biased and Unbiased Specular Manifold Sampling, primarily for **refractive caustics**.

Key features:

*   Core MNEE functions (`integrator/mnee.h`) updated, e.g., with an explicit (but SMS-unused for true) `reflection` parameter. Can use **angle difference constraints** for the refractive Newton solver (UI choice).
*   New `integrator/sms.h` with SMS algorithms:
    *   `sms_find_caster_chain` finds **refractive/glass** multi-scatter paths.
    *   `integrate_sms_unbiased` and `integrate_sms_biased` for these chains.
*   Stochastic initialization (random barycentrics + random microfacet normals).
*   Integration into `integrator/shade_surface.h` to replace NEE if SMS succeeds.

Reflection support is present in core functions but disabled in the SMS pipeline due to solver issues. Angle difference constraints are partially supported for the refractive solver but are flawed. Future work could target robust reflection, full angle difference integration (including MIS), and other SMS paper features.