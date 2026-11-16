"""Explicit numerical adaptation for offline native-VBD references only.

Newton's VBD already calls the shared maximal-coordinate mimic projection.
This adapter exposes relaxation and an explicit projection schedule while
preserving the constraint, masses and native kernel formulas. It does not
alter Newton files and must never be installed in the learned solver runtime.
"""
import math


class ReferenceMimicRelaxation:
    def __init__(self, relaxation=1., *, projection_interval=1):
        if not math.isfinite(relaxation) or not 0 < relaxation <= 1:
            raise ValueError('Reference mimic relaxation must be in (0, 1]')
        self.relaxation = relaxation
        if not isinstance(projection_interval,int) or projection_interval<1:
            raise ValueError('A positive fixed projection interval is required')
        self.projection_interval = projection_interval
        self.calls = 0
        self.opportunities = 0
        self._module = None
        self._original = None

    def install(self):
        if self._module is not None: raise RuntimeError('Reference adapter already installed')
        if self.relaxation == 1. and self.projection_interval==1: return self
        import warp as wp
        from newton._src.solvers.vbd import solver_vbd
        from newton._src.solvers.xpbd import kernels
        self._module = solver_vbd
        self._original = solver_vbd.project_joint_mimics

        def project(model, body_q, body_qd, inverse_mass, inverse_inertia, deltas, dt):
            self.opportunities += 1
            if self.opportunities % self.projection_interval: return
            self.calls += 1
            if self.relaxation==1.:
                return self._original(model,body_q,body_qd,inverse_mass,inverse_inertia,deltas,dt)
            deltas.zero_()
            wp.launch(kernels.solve_joint_mimics, dim=model.joint_count,
                inputs=[body_q,model.body_com,inverse_mass,inverse_inertia,
                        model.joint_type,model.joint_enabled,model.joint_parent,model.joint_child,
                        model.joint_X_p,model.joint_X_c,model.joint_qd_start,model.joint_dof_dim,
                        model.joint_axis,model.joint_mimic_joint,model.joint_mimic_coeffs,
                        self.relaxation,self.relaxation,dt],
                outputs=[deltas,None],device=model.device)
            wp.launch(kernels.apply_joint_mimic_deltas, dim=model.body_count,
                inputs=[body_q,body_qd,model.body_com,inverse_mass,inverse_inertia,deltas,dt],device=model.device)

        self._project = project
        solver_vbd.project_joint_mimics = project
        return self

    def restore(self):
        if self._module is not None:
            self._module.project_joint_mimics = self._original
            self._module = None

    def __enter__(self): return self.install()
    def __exit__(self, *unused): self.restore()
