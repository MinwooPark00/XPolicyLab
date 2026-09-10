from enum import Enum


"""
Embodiment tags are used to identify the robot embodiment in the data.

Naming convention:
<dataset>_<robot_name>

If using multiple datasets, e.g. sim GR1 and real GR1, we can drop the dataset name and use only the robot name.
"""


class EmbodimentTag(Enum):
    ##### Pretrain embodiment tags #####
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    """
    The RoboCasa Panda robot with omron mobile base.
    """

    GR1 = "gr1"
    """
    The Fourier GR1 robot.
    """

    ##### Pre-registered posttrain embodiment tags #####
    UNITREE_G1 = "unitree_g1"
    """
    The Unitree G1 robot.
    """

    LIBERO_PANDA = "libero_panda"
    """
    The Libero panda robot.
    """

    OXE_GOOGLE = "oxe_google"
    """
    The Open-X-Embodiment Google robot.
    """

    OXE_WIDOWX = "oxe_widowx"
    """
    The Open-X-Embodiment WidowX robot.
    """

    BEHAVIOR_R1_PRO = "behavior_r1_pro"
    """
    The Behavior R1 Pro robot.
    """

    G1_EE_A16 = "g1_ee_a16"
    """
    G1 end-effector pretraining embodiment with action horizon 16.
    """

    H1_EE_A16 = "h1_ee_a16"
    """
    H1 end-effector pretraining embodiment with action horizon 16.
    """

    G1_LOCO_DOWNSTREAM = "g1_loco_downstream"
    """
    G1 locomotion downstream fine-tuning embodiment.
    """

    G1_UPPER_A16 = "g1_upper_a16"
    """
    G1 upper-body manipulation pretraining embodiment with action horizon 16.
    """

    # New embodiment during post-training
    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment.
    """
