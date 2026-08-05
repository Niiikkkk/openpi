"""Pure-USD/PhysX-schema reimplementations of the patches IsaacLab's spawner applies when this
scene is referenced via `SubPrimReferenceCfg` (see
`ur7e_bimanual_pick/tasks/spawners.py::spawn_sub_prim_reference`).

A raw `open_stage(usd_path)` (as in `runner_single.py`/`sim_state.py`) never goes through that
spawner, so none of these patches get applied -- the robot ends up with PhysX's stock defaults
instead. These functions only use `pxr` (Usd/UsdPhysics/PhysxSchema), Omniverse's own USD Python
bindings, not `isaaclab` -- so they work in an environment that has `isaacsim` but not `isaaclab`.

Ported near-verbatim from `spawners.py`; see that file's docstrings for the full reasoning behind
each patch. Call `patch_robot_prim(prim)` on the robot's root prim once, right after the stage is
loaded and before `Articulation.initialize()` -- solver iteration counts and mimic joints are
schema authored on the stage, and should be in place before PhysX parses/cooks the articulation.
"""

import math

from pxr import PhysxSchema, Usd, UsdPhysics


def patch_robot_prim(prim: Usd.Prim) -> None:
    """Apply all three patches to `prim`'s subtree. Safe to call more than once (each patch is
    idempotent / a no-op against a prim that's already been patched or doesn't need it).
    """
    _dedupe_nested_articulation_roots(prim)
    _raise_articulation_solver_iterations(prim)
    _add_gripper_mimic_joints(prim)


def _dedupe_nested_articulation_roots(prim: Usd.Prim) -> None:
    """Strip ArticulationRootAPI from any prim nested inside another prim that already has it.

    See `spawners.py::_dedupe_nested_articulation_roots` for the full explanation: the source USD's
    URDF-import pipeline leaves a `Geometry` (visual-only) subtree also tagged as an articulation
    root, so PhysX cooks it as a second, disconnected articulation -- its meshes then never follow
    the real `Physics` joints.
    """
    roots = [descendant for descendant in Usd.PrimRange(prim) if descendant.HasAPI(UsdPhysics.ArticulationRootAPI)]
    root_paths = [root.GetPath() for root in roots]
    for root, root_path in zip(roots, root_paths):
        if any(other != root_path and root_path.HasPrefix(other) for other in root_paths):
            root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            root.RemoveAppliedSchema("NewtonArticulationRootAPI")


def _raise_articulation_solver_iterations(prim: Usd.Prim) -> None:
    """Raise PhysX solver iteration counts on any articulation root under `prim`.

    The source USD has no authored `physxArticulation:solverPositionIterationCount`, so PhysX falls
    back to its default (4 position / 1 velocity iteration) -- too few for a long, stiff 6-DOF
    serial chain to actually reach the PD spring's theoretical equilibrium each substep. This is
    important to apply here: under-convergence looks identical to "not enough stiffness" (continued
    sag under gravity even at very high gains) from the outside, so without this patch, a raw-stage
    robot will show *worse* sag than the properly-configured data-collection env even when given the
    exact same stiffness/damping values.
    """
    for descendant in Usd.PrimRange(prim):
        if descendant.HasAPI(UsdPhysics.ArticulationRootAPI):
            physx_api = PhysxSchema.PhysxArticulationAPI.Apply(descendant)
            physx_api.GetSolverPositionIterationCountAttr().Set(32)
            physx_api.GetSolverVelocityIterationCountAttr().Set(4)


def _add_gripper_mimic_joints(prim: Usd.Prim) -> None:
    """Add a PhysxMimicJointAPI (tied to finger_joint) to any passive gripper linkage joint missing
    one, for any gripper root whose name contains "Robotiq_2F_85" under `prim`.

    Without this, the gripper's passive linkage joints aren't coupled to `finger_joint` at all --
    the mechanism isn't properly constrained, which is the most likely cause of inconsistent/
    premature gripper closing seen when driving a raw-stage robot. See
    `spawners.py::_add_gripper_mimic_joints` for the full reasoning (including the loop-closure-joint
    exception this respects -- some asset variants already have a real loop-closing joint, in which
    case adding synthetic mimics on top over-constrains the mechanism and locks it solid).

    Confirmed empirically that this asset genuinely has a real loop-closure joint (PhysX's own
    cook-time redundant-DOF detection auto-excludes one passive joint from the articulation, even
    though no joint has `physxJoint:excludeFromArticulation` authored beforehand to check for it) --
    an attempt to force mimics onto the *other* passive joints anyway (on the theory that the
    infinite-limit heuristic below was a false positive) locked the whole mechanism solid
    (`finger_joint` stopped moving at all), exactly the failure mode this skip condition exists to
    prevent. So the infinite-limit heuristic, imperfect as a loop-closure detector as it is, is the
    correct behavior to keep here -- do not remove it again without first finding an actual way to
    detect PhysX's cook-time exclusion instead of guessing at additional mimics.
    """

    def has_finite_limits(joint_prim: Usd.Prim) -> bool:
        lower = joint_prim.GetAttribute("physics:lowerLimit")
        upper = joint_prim.GetAttribute("physics:upperLimit")
        if not lower.IsValid() or not upper.IsValid():
            return False
        return math.isfinite(lower.Get()) and math.isfinite(upper.Get())

    for descendant in Usd.PrimRange(prim):
        if "Robotiq_2F_85" not in descendant.GetName():
            continue
        finger_joint = None
        passive_joints = []
        for gripper_descendant in Usd.PrimRange(descendant):
            if gripper_descendant.GetTypeName() != "PhysicsRevoluteJoint":
                continue
            if gripper_descendant.GetName() == "finger_joint":
                finger_joint = gripper_descendant
            else:
                passive_joints.append(gripper_descendant)
        if finger_joint is None:
            continue
        if any(not has_finite_limits(joint_prim) for joint_prim in passive_joints):
            continue
        for joint_prim in passive_joints:
            if joint_prim.HasAPI(PhysxSchema.PhysxMimicJointAPI):
                continue
            mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(joint_prim, "rotX")
            mimic_api.GetReferenceJointRel().SetTargets([finger_joint.GetPath()])
            mimic_api.GetGearingAttr().Set(-1.0)
            mimic_api.GetOffsetAttr().Set(0.0)
            mimic_api.GetDampingRatioAttr().Set(0.01)
            mimic_api.GetNaturalFrequencyAttr().Set(5000.0)
