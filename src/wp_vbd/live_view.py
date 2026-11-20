"""Optional Newton window that displays accepted inference states only."""


class LiveView:
    def __init__(self,model,*,enabled=False,position=(.8,-1.1,.9),target=(0.,-.4,.3)):
        self.viewer=None
        if enabled:
            import numpy as np
            import warp as wp
            from newton.viewer import ViewerGL
            self.viewer=ViewerGL(width=1440,height=900,vsync=False)
            self.viewer.set_model(model)
            self.viewer.set_camera(wp.vec3(*position))
            self.viewer.camera.look_at(np.asarray(target))

    def present(self,state,physical_time):
        if self.viewer is None:return True
        self.viewer.begin_frame(physical_time)
        self.viewer.log_state(state)
        self.viewer.end_frame()
        return self.viewer.is_running()

    def close(self):
        if self.viewer is not None:self.viewer.close()
