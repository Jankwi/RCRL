LOCAL_BINDINGS="configs/lb.gin"

CONFIG=$1

uv run runner.py --config_file $CONFIG --binding_files $LOCAL_BINDINGS