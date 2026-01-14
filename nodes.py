
import torch
import numpy as np
JOINT_NAMES = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", 
    "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip", 
    "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
]

LIMB_SEQ = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

JOINT_PARENTS = {
    0: 1, 1: -1, 2: 1, 3: 2, 4: 3, 5: 1, 6: 5, 7: 6, 8: 1, 9: 8, 10: 9,
    11: 1, 12: 11, 13: 12, 14: 0, 15: 0, 16: 14, 17: 15,
}

BONE_COLORS = [
    [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
    [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
    [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
    [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
    [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
    [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
]


class HYMotionToSCAILBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA",),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 64}),
                # 核心修正：默认缩放改为 300，使其匹配 SCAIL 的画面
                "scale": ("FLOAT", {"default": 300.0, "min": 10.0, "max": 1000.0, "step": 10.0}),
                # 核心修正：增加 X 和 Y 偏移，方便居中
                "offset_x": ("FLOAT", {"default": 0.0, "min": -2000.0, "max": 2000.0}),
                "offset_y": ("FLOAT", {"default": 100.0, "min": -2000.0, "max": 2000.0}),
                "offset_z": ("FLOAT", {"default": 800.0, "min": 0.0, "max": 5000.0}),
            },
        }

    RETURN_TYPES = ("SCAIL_POSE_SEQUENCE",)
    RETURN_NAMES = ("pose_sequence",)
    FUNCTION = "convert"
    CATEGORY = "HY-Motion/Bridge"

    def convert(self, motion_data, sample_index, scale, offset_x, offset_y, offset_z):
        # 1. 获取数据
        kpts = motion_data.output_dict.get("keypoints3d")
        if kpts is None:
            raise ValueError("No keypoints3d in motion_data")

        idx = min(sample_index, kpts.shape[0] - 1)
        src_poses = kpts[idx].cpu().numpy()  # [Frames, 22, 3]
        num_frames = src_poses.shape[0]

        # 2. 映射表 (SMPL 22 -> OpenPose 18)
        # SMPL: 0:Pelvis, 1:L_Hip, 2:R_Hip, 3:Spine1, 6:Spine2, 9:Spine3, 12:Neck, 15:Head
        #       16:L_Collar, 17:R_Collar, 18:L_Elbow, 19:R_Elbow, 20:L_Wrist, 21:R_Wrist
        #       4:L_Knee, 5:R_Knee, 7:L_Ankle, 8:R_Ankle
        
        mapping = {
            1: 12,  # Neck
            2: 17,  # R_Shoulder
            3: 19,  # R_Elbow
            4: 21,  # R_Wrist
            5: 16,  # L_Shoulder
            6: 18,  # L_Elbow
            7: 20,  # L_Wrist
            8: 2,   # R_Hip
            9: 5,   # R_Knee
            10: 8,  # R_Ankle
            11: 1,  # L_Hip
            12: 4,  # L_Knee
            13: 7,  # L_Ankle
            0: 15,  # Nose (用 Head 代替)
        }

        scail_poses = []

        for f in range(num_frames):
            src = src_poses[f]
            # 初始化目标 (18, 3)
            op18 = np.zeros((18, 3), dtype=np.float32)

            # --- 步骤 A: 映射关节 ---
            for target_idx, src_idx in mapping.items():
                if src_idx < len(src):
                    op18[target_idx] = src[src_idx]

            # --- 步骤 B: 坐标系修正 (关键！) ---
            # 1. 翻转 Y 轴 (HY-Motion Y-up -> SCAIL Y-down)
            op18[:, 1] = -op18[:, 1]

            # 2. 缩放 (把 1.8米 变成 ~540 像素单位)
            op18 *= scale

            # 3. 居中校正
            # HY-Motion 的 Pelvis(0) 通常在 (0,0,0)。
            # SCAIL 渲染器的 (0,0) 是屏幕中心。
            # 我们需要把人往下移一点，不然脚会飘在屏幕外。
            op18[:, 0] += offset_x
            op18[:, 1] += offset_y 
            op18[:, 2] += offset_z

            # --- 步骤 C: 伪造头部/五官 (让头看起来是个圆圈而不是一个点) ---
            head_pos = op18[0] # Nose 位置
            # 这里的偏移量也要基于 scale 来计算，不然头还是一个点
            head_scale = scale * 0.05 
            
            op18[14] = head_pos + np.array([-head_scale, -head_scale, 0]) # R_Eye
            op18[15] = head_pos + np.array([ head_scale, -head_scale, 0]) # L_Eye
            op18[16] = head_pos + np.array([-head_scale*2, 0, head_scale]) # R_Ear
            op18[17] = head_pos + np.array([ head_scale*2, 0, head_scale]) # L_Ear

            scail_poses.append(op18)

        return ({
            "poses": scail_poses,
            "limb_seq": LIMB_SEQ,
            "bone_colors": BONE_COLORS,
            "char_count": 1 # 告知 SCAIL 这是一个单人动作
        },)
# 记得在 NODE_CLASS_MAPPINGS 中注册

# ============================================================================
# Node: HYMotion to NLF Bridge V6 (With Head/Face Projection)
# ============================================================================

class HYMotionToNLFBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA",),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 64}),
                "scale": ("FLOAT", {"default": 1000.0, "min": 1.0, "max": 2000.0, "step": 10.0, "tooltip": "1000 converts meters to mm."}),
                "z_offset": ("FLOAT", {"default": 2500.0, "min": 0.0, "max": 10000.0, "step": 100.0, "tooltip": "Distance in mm. Determines perspective strength."}),
                "auto_center": ("BOOLEAN", {"default": True, "tooltip": "Center the first frame at (0,0,0)"}),
                "rotation_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 5.0, "tooltip": "0 = Face Camera. Rotates the 3D model."}),
            },
            "optional": {
                "width": ("INT", {"default": 512, "tooltip": "Canvas width for 2D projection"}),
                "height": ("INT", {"default": 896, "tooltip": "Canvas height for 2D projection"}),
            }
        }

    RETURN_TYPES = ("NLFPRED", "DWPOSES")
    RETURN_NAMES = ("nlf_poses", "projected_dw_poses")
    FUNCTION = "convert"
    CATEGORY = "HY-Motion/Bridge"

    def convert(self, motion_data, sample_index, scale, z_offset, auto_center, rotation_y, width=512, height=896):
        import torch
        import numpy as np

        kpts = motion_data.output_dict.get("keypoints3d")
        if kpts is None:
            raise ValueError("No keypoints3d found in motion data")
        
        idx = min(sample_index, kpts.shape[0] - 1)
        poses = kpts[idx].clone() # [Frames, 22, 3]
        
        # --- 1. 3D Processing ---
        
        if auto_center:
            root_pos = poses[0, 0, :].clone()
            poses = poses - root_pos

        poses *= scale
        
        # Apply Rotation (Y-axis)
        # SMPL defaults facing +Z. We want 0 deg to face camera (-Z).
        theta_deg = rotation_y + 180.0
        theta_rad = np.radians(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        
        x = poses[:, :, 0].clone()
        z = poses[:, :, 2].clone()
        poses[:, :, 0] = x * cos_t - z * sin_t
        poses[:, :, 2] = x * sin_t + z * cos_t
        
        # Coordinate Conversion (Y-up -> Y-down)
        poses[:, :, 1] = -poses[:, :, 1]
        
        # Z Offset
        poses[:, :, 2] += z_offset
        
        # Padding for NLF (22 -> 24)
        frames, num_joints, dims = poses.shape
        if num_joints == 22:
            padding = torch.zeros((frames, 2, dims), device=poses.device, dtype=poses.dtype)
            padding[:, 0, :] = poses[:, 20, :] # L_Hand from L_Wrist
            padding[:, 1, :] = poses[:, 21, :] # R_Hand from R_Wrist
            poses_final = torch.cat([poses, padding], dim=1)
        else:
            poses_final = poses

        # Pack NLF Output
        joints_list = []
        for i in range(frames):
            frame_pose = poses_final[i].unsqueeze(0)
            joints_list.append(frame_pose)
        nlf_output = {'joints3d_nonparam': [joints_list]}

        # --- 2. Generate Projected DWPose (With FACE) ---
        
        # Mapping SMPL(22) to OpenPose(18)
        # SMPL: 0:Pelvis, 1:L_Hip, 2:R_Hip, 3:Spine1, 4:L_Knee, 5:R_Knee, 6:Spine2, 7:L_Ank, 8:R_Ank, 9:Spine3, 
        #       10:L_Foot, 11:R_Foot, 12:Neck, 13:L_Collar, 14:R_Collar, 15:Head, 16:L_Sho, 17:R_Sho, 
        #       18:L_Elb, 19:R_Elb, 20:L_Wri, 21:R_Wri
        
        # OpenPose 18: 0:Nose, 1:Neck, 2:RSho, 3:RElb, 4:RWri, 5:LSho, 6:LElb, 7:LWri, 
        #              8:RHip, 9:RKnee, 10:RAnk, 11:LHip, 12:LKnee, 13:LAnk, 
        #              14:REye, 15:LEye, 16:REar, 17:LEar
        
        # Simple pinhole projection
        focal = max(width, height) # Approximate focal length
        cx, cy = width / 2, height / 2
        
        def project(p3d):
            # p3d: [..., 3] (x, y, z)
            x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
            # Avoid div by zero
            z = torch.clamp(z, min=10.0)
            u = (x / z) * focal + cx
            v = (y / z) * focal + cy
            return torch.stack([u, v], dim=-1)

        # Generate fake face points in 3D relative to Head(15)
        # We assume poses is currently in Camera Space (Y-down, X-right)
        # Head is roughly at center.
        head_3d = poses[:, 15, :]
        
        # Define offsets (in mm)
        face_scale = scale * 0.06 # Adjust based on body scale
        
        # Note: In screen view (viewer looking at screen):
        # Right Eye (User's Right) is +X, Left Eye is -X. 
        # But OpenPose defines R_Eye as the person's Right Eye.
        # If person faces camera: Person's Right Eye is on Viewer's Left (-X).
        
        # 3D Offsets
        off_nose   = torch.tensor([0.0, 0.0, -face_scale*0.5], device=poses.device) # Slightly forward (negative Z is closer)
        off_r_eye  = torch.tensor([-face_scale*0.6, -face_scale*0.3, 0.0], device=poses.device)
        off_l_eye  = torch.tensor([ face_scale*0.6, -face_scale*0.3, 0.0], device=poses.device)
        off_r_ear  = torch.tensor([-face_scale*1.5, 0.0, face_scale*0.5], device=poses.device)
        off_l_ear  = torch.tensor([ face_scale*1.5, 0.0, face_scale*0.5], device=poses.device)
        
        # Apply offsets to all frames
        nose_3d = head_3d + off_nose
        r_eye_3d = head_3d + off_r_eye
        l_eye_3d = head_3d + off_l_eye
        r_ear_3d = head_3d + off_r_ear
        l_ear_3d = head_3d + off_l_ear
        
        # Construct full 18-point skeleton in 3D first
        frames_cnt = poses.shape[0]
        op_3d = torch.zeros((frames_cnt, 18, 3), device=poses.device)
        
        op_3d[:, 0] = nose_3d
        op_3d[:, 1] = poses[:, 12] # Neck
        op_3d[:, 2] = poses[:, 17] # R_Sho
        op_3d[:, 3] = poses[:, 19] # R_Elb
        op_3d[:, 4] = poses[:, 21] # R_Wri
        op_3d[:, 5] = poses[:, 16] # L_Sho
        op_3d[:, 6] = poses[:, 18] # L_Elb
        op_3d[:, 7] = poses[:, 20] # L_Wri
        op_3d[:, 8] = poses[:, 2]  # R_Hip
        op_3d[:, 9] = poses[:, 5]  # R_Knee
        op_3d[:, 10] = poses[:, 8] # R_Ank
        op_3d[:, 11] = poses[:, 1] # L_Hip
        op_3d[:, 12] = poses[:, 4] # L_Knee
        op_3d[:, 13] = poses[:, 7] # L_Ank
        op_3d[:, 14] = r_eye_3d
        op_3d[:, 15] = l_eye_3d
        op_3d[:, 16] = r_ear_3d
        op_3d[:, 17] = l_ear_3d
        
        # Project to 2D
        op_2d = project(op_3d) # [Frames, 18, 2] in pixels
        
        # Normalize to 0.0 - 1.0 (DWPose Standard)
        op_2d[:, :, 0] /= width
        op_2d[:, :, 1] /= height
        
        # Convert to numpy list structure for DWPose
        op_2d_np = op_2d.cpu().numpy()
        
        dw_list = []
        subset_base = np.arange(18, dtype=np.float32)
        
        for i in range(frames_cnt):
            candidate = op_2d_np[i] # [18, 2]
            
            # Format: 'bodies': {'candidate': [N, 18, 2], 'subset': [N, 18]}
            # N=1 person
            body_entry = {
                'candidate': candidate[None, ...], # [1, 18, 2]
                'subset': subset_base[None, ...]   # [1, 18]
            }
            
            frame_dict = {
                'bodies': body_entry,
                'hands': np.zeros((1, 21, 2), dtype=np.float32), # Optional: could project hands too
                'faces': np.zeros((1, 68, 2), dtype=np.float32),
                'canvas_width': width,
                'canvas_height': height
            }
            dw_list.append(frame_dict)
            
        dummy_dw_poses = {"poses": dw_list, "swap_hands": False}
        
        return (nlf_output, dummy_dw_poses)
# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "HYMotionToSCAILBridge":HYMotionToSCAILBridge,
    "HYMotionToNLFBridge":HYMotionToNLFBridge,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotionToSCAILBridge":"HYMotion To SCAILBridge",
    "HYMotionToNLFBridge":"HYMotion To NLF Bridge",

}
    

