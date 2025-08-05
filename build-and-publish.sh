#!/bin/bash

docker buildx build \                                              
  --platform linux/amd64 \
  -t owengombas/dgan:amd64 \
  --push .
