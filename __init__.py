from .nodes import HYMotionToSCAILBridge, HYMotionToNLFBridge

NODE_CLASS_MAPPINGS = {
    "HYMotionToSCAILBridge":HYMotionToSCAILBridge,
    "HYMotionToNLFBridge":HYMotionToNLFBridge,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotionToSCAILBridge":"HYMotion To SCAILBridge",
    "HYMotionToNLFBridge":"HYMotion To NLF Bridge",

}
    

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]