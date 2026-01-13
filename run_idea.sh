#!/bin/bash

TOPIC="Pointmap Representation (W x H x 3grid), Unposed & Uncalibrated Dense 3D Reconstruction, 3D Scene Flow & Motion Estimation"

read -r -d '' BACKGROUND <<'EOF'
Traditional 3D reconstruction methods are often limited by the need for pre-calibrated camera parameters and static scene assumptions, which are difficult to satisfy in dynamic environments like surgical endoscopy. To overcome this, the Pointmap Representation ($W \times H \times 3$ grid) has emerged as a unified approach to store dense 3D coordinates for every pixel in a common coordinate frame, effectively solving for geometry, correspondences, and relative poses simultaneously. This foundation was established by models like DUSt3R, which enables Unposed & Uncalibrated Dense 3D Reconstruction from arbitrary image pairs. These systems often leverage pre-trained foundation models such as CroCo (Cross-View Completion)—a self-supervised scheme similar to Masked Autoencoders (MAE)—to initialize robust cross-view features7. Building on this, Endo3R introduces 3D Scene Flow & Motion Estimation to handle non-rigid tissue deformation and tool movement by calculating 3D displacements between sequential pointmaps. By utilizing a DPT-style dense regression head and a Dynamics-aware Flow Loss, the framework optimizes its internal "3D world model" by aligning projected motion with observed 2D optical flow from off-the-shelf networks, thereby bypassing the need for expensive ground-truth depth or pose labels.
EOF

# Check if arguments are provided
if [ "$#" -ge 1 ]; then
    TOPIC="$1"
fi

if [ "$#" -ge 2 ]; then
    BACKGROUND="$2"
fi


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ENV_FILE="$SCRIPT_DIR/src/agents/idea_agent/env/env.sh"
if [ -f "$ENV_FILE" ]; then
    echo "Sourcing environment variables from $ENV_FILE"
    source "$ENV_FILE"
fi

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

python "$SCRIPT_DIR/src/agents/idea_agent/main.py" --topic "$TOPIC" --background_knowledge "$BACKGROUND"
