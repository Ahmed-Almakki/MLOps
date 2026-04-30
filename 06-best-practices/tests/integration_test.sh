#!/usr/bin/env bash

# exit immedialty if a command exits with a none-zero status
set -e

echo "Starting localStack..."

docker compose up -d

# sleep for 5 sec until localstack begin
sleep 5

export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
export S3_ENDPOINT_URL="http://localhost:4566"

aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration

echo "Starting Intergration test..."

# (Disabling "set -e" temporarily so we can capture the test exit code)
set +e
