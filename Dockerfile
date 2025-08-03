FROM --platform=$BUILDPLATFORM python:slim

ENV RANKS=
ENV MASTER_ADDR=
ENV WORLD_SIZE=

COPY src/requirements.txt .
RUN pip3 install --user -r requirements.txt

COPY shared-args.sh .
COPY run-distributed.sh .
COPY run-standalone.sh .
RUN chmod +x *.sh

COPY src .

CMD ["bash", "-c", "bash run-distributed.sh \"$RANK\" \"$MASTER_ADDR\" \"$WORLD_SIZE\""]
