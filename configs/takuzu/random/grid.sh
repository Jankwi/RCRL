
declare -A dataset_class=(
    ["relative"]="RelativeContrastiveDataset"
    ["crtr"]="ContrastiveDatasetNoSpecial"
    ["crl"]="ContrastiveDatasetNoSpecial"
)


for method in relative crtr crl; do
    for geom_type in bounded clamped; do
        echo "Running with method=$method, geom_type=$geom_type"
        uv run runner.py --config_file configs/takuzu/random/${method}.gin --gin_bindings "${dataset_class[$method]}.geom_type='${geom_type}'"
    done
done