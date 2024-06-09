FROM python:3.9-slim

ARG MODEL_PATH

RUN mkdir /home/talkaa
ENV PROJECT_DIR /home/talkaa
ENV MODEL_FILEPATH $MODEL_PATH
WORKDIR /home/talkaa


RUN echo "The value of MODEL_FILEPATH is: $MODEL_PATH"

COPY requirments.txt /${PROJECT_DIR}/requirements.txt
RUN apt-get -y update && \
    apt-get -y install apt-utils gcc curl && \
    pip install --no-cache-dir -r requirements.txt

COPY src /${PROJECT_DIR}/src


ADD run_inf_server.sh /${PROJECT_DIR}/run_inf_server.sh
RUN chmod +x /${PROJECT_DIR}/run_inf_server.sh
CMD [ "./run_inf_server.sh" ]
