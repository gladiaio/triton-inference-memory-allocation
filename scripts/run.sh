function isready() {
    if curl --silent --fail localhost:$1/v2/health/ready 2> /dev/null;
    then
        return 1
    else
        return 0

    fi
}

function waituntil() {
    while [[ 0 -eq 0 ]]
    do
        if ! isready $1
            then
                break
            fi
    done
}

# remove if already running
docker compose -f docker/docker-compose.yaml down --remove-orphans

# setup
pip install -r requirements.txt
python scripts/create_dummy_onnx.py

# Run for pyannote
docker compose -f docker/docker-compose.yaml up --build --detach triton-server-pyannote
waituntil 8000
nvidia-smi -f pyannote.ready.gpu
python scripts/calls.py --test pyannote
nvidia-smi -f pyannote.after-test.gpu

# Run for dummy ("memory leak")
docker compose -f docker/docker-compose.yaml down --remove-orphans
docker compose -f docker/docker-compose.yaml up --build --detach triton-server-dummy
waituntil 8000
nvidia-smi -f dummy.ready.gpu
python scripts/calls.py --test dummy
nvidia-smi -f dummy.after-test.gpu


# THIS CODE IS NOT YET REPRODUCING OUR PROBLEMS
# # Run for MFCC
# docker compose -f docker/docker-compose.yaml down --remove-orphans
# docker compose -f docker/docker-compose.yaml up --build --detach triton-server-mel
# waituntil 8000
# nvidia-smi -f mel.ready.gpu
# python scripts/calls.py --test mel
# nvidia-smi -f mel.after-test.gpu