# $RANDOM gives a random number to avoid name conflicts
docker run \
    --name="example-$RANDOM" \
    --mount type=bind,source="$(pwd)"/container/app/data,target=/app/data \
    --entrypoint python3 \
    example \
    -m app.train