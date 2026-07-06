#!/usr/bin/env bash
# ==============================================================================
# run_local.sh - Local Training & Demo Runner for OpenArm A1
# ==============================================================================
ISAAC_SIM_PYTHON="${ISAAC_SIM_PYTHON:-/home/hans/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh}"

function print_usage() {
    echo "Usage: ./run_local.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  train       Start local training using settings optimized for local GPU (RTX 4050)"
    echo "  demo        Run visual playback/evaluation of a trained local model"
    echo "  help        Show this help message"
    echo ""
    echo "Options for 'train':"
    echo "  --envs <num>      Number of parallel environments (default: 16)"
    echo "  --steps <num>     Total training timesteps (default: 1000000)"
    echo "  --gui             Run training with Isaac Sim GUI window visible (default: headless)"
    echo "  --resume          Resume training from the latest checkpoint"
    echo "  --progress        Enable tqdm progress bar"
    echo "  --phase <1|2>     Task phase: 1=reach, 2=reach+grasp+lift"
    echo "  --stage <s>       Phase 2 sub-stage: reach | grasp | lift | all (default: all)"
    echo "  --assist-schedule Phase 2: decay assist 1.0→0.0 during training"
    echo "  --checkpoint <path> Fine-tune from .pt weights"
    echo ""
    echo "Options for 'demo':"
    echo "  --model <path>    Path to model weights/checkpoint (default: ./logs/active_policy.pt)"
    echo "  --envs <num>      Number of parallel demo environments (default: 1)"
    echo "  --phase <1|2>     Task phase: 1=reach, 2=reach+grasp+lift"
    echo "  --stage <s>       Phase 2 sub-stage: reach | grasp | lift | all (default: all)"
    echo "  --assist          Same as --stage all (legacy)"
}

# Ensure script is run from workspace root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

CMD=$1
shift

case "$CMD" in
    train)
        ENVS=16
        STEPS=1000000
        HEADLESS="--headless"
        RESUME=""
        PROGRESS=""
        LOG_DIR="./logs/train"
        TASK_PHASE=""
        ASSIST_SCHEDULE=""
        CHECKPOINT=""
        STAGE=""

        while [ $# -gt 0 ]; do
            case "$1" in
                --envs)
                    ENVS=$2
                    shift 2
                    ;;
                --steps)
                    STEPS=$2
                    shift 2
                    ;;
                --gui)
                    HEADLESS=""
                    shift
                    ;;
                --resume)
                    RESUME="--resume"
                    shift
                    ;;
                --progress)
                    PROGRESS="--progress"
                    shift
                    ;;
                --phase)
                    TASK_PHASE="--task_phase $2"
                    shift 2
                    ;;
                --stage)
                    STAGE="--stage $2"
                    shift 2
                    ;;
                --assist-schedule)
                    ASSIST_SCHEDULE="--assist-schedule"
                    shift
                    ;;
                --checkpoint)
                    CHECKPOINT="--checkpoint $2"
                    shift 2
                    ;;
                *)
                    echo "Unknown training option: $1"
                    print_usage
                    exit 1
                    ;;
            esac
        done

        echo "======================================================================"
        echo "Starting Local OpenArm PPO Training (Manager-Based RL)"
        echo "----------------------------------------------------------------------"
        echo "  GPU Device : NVIDIA GeForce RTX 4050 (6GB VRAM)"
        echo "  Environments: $ENVS parallel"
        echo "  Timesteps   : $STEPS"
        echo "  Mode        : $( [ -z "$HEADLESS" ] && echo "GUI" || echo "Headless" )"
        echo "  Resume      : $( [ -n "$RESUME" ] && echo "Yes" || echo "No" )"
        echo "  Progress    : $( [ -n "$PROGRESS" ] && echo "Enabled (tqdm)" || echo "Disabled" )"
        echo "  Log Dir     : $LOG_DIR"
        echo "======================================================================"

        PYTHONPATH="$SCRIPT_DIR" $ISAAC_SIM_PYTHON ./isaaclab_train.py \
            --num_envs "$ENVS" \
            --timesteps "$STEPS" \
            --log_dir "$LOG_DIR" \
            $HEADLESS \
            $RESUME \
            $PROGRESS \
            $TASK_PHASE \
            $STAGE \
            $ASSIST_SCHEDULE \
            $CHECKPOINT
        ;;

    demo)
        MODEL="./logs/active_policy.pt"
        ENVS=1
        EXTRA_ARGS=()
        STAGE=""

        while [ $# -gt 0 ]; do
            case "$1" in
                --model)
                    MODEL=$2
                    shift 2
                    ;;
                --envs)
                    ENVS=$2
                    shift 2
                    ;;
                --phase)
                    EXTRA_ARGS+=("--task_phase" "$2")
                    shift 2
                    ;;
                --stage)
                    STAGE="$2"
                    shift 2
                    ;;
                --assist)
                    STAGE="all"
                    shift
                    ;;
                *)
                    EXTRA_ARGS+=("$1")
                    shift
                    ;;
            esac
        done

        if [ -n "$STAGE" ]; then
            EXTRA_ARGS+=("--stage" "$STAGE")
        fi

        echo "======================================================================"
        echo "Starting Visual Playback (Demo) of Local Model"
        echo "----------------------------------------------------------------------"
        echo "  Model Path : $MODEL"
        echo "  Environments: $ENVS"
        echo "======================================================================"

        PYTHONPATH="$SCRIPT_DIR" $ISAAC_SIM_PYTHON ./isaaclab_demo.py \
            --model_path "$MODEL" \
            --num_envs "$ENVS" \
            --no-bottle-rand \
            "${EXTRA_ARGS[@]}"
        ;;

    help)
        print_usage
        ;;

    *)
        echo "Unknown command: $CMD"
        print_usage
        exit 1
        ;;
esac
