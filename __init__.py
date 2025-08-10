from .py.motionConfig import MotionConfigNode
from .py.motionPosition import MotionPositionNode
from .py.motionRotation import MotionRotationNode
from .py.motionScale import MotionScaleNode
from .py.motionOpacity import MotionOpacityNode
from .py.motionPositionOnPath import MotionPositionOnPathNode
from .py.motionDistortion import MotionDistortionNode
from .py.motionShake import MotionShakeNode
from .py.motionMask import MotionMaskNode
import os

NODE_CLASS_MAPPINGS = {
    "💀Motion Config": MotionConfigNode,
    "💀Motion Position": MotionPositionNode,
    "💀Motion Rotation": MotionRotationNode,
    "💀Motion Scale": MotionScaleNode,
    "💀Motion Opacity": MotionOpacityNode,
    "💀Motion Position On Path": MotionPositionOnPathNode,
    "💀Motion Distortion": MotionDistortionNode,
    "💀Motion Shake": MotionShakeNode,
    "💀Motion Mask": MotionMaskNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "💀Motion Config": "💀Motion Config",
    "💀Motion Position": "💀Motion Position",
    "💀Motion Rotation": "💀Motion Rotation",
    "💀Motion Scale": "💀Motion Scale",
    "💀Motion Opacity": "💀Motion Opacity",
    "💀Motion Position On Path": "💀Motion Position On Path",
    "💀Motion Distortion": "💀Motion Distortion",
    "💀Motion Shake": "💀Motion Shake",
    "💀Motion Mask": "💀Motion Mask",
}
