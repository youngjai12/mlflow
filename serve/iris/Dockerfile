FROM python:3.9-slim

RUN mkdir /home/talkaa
ENV PROJECT_DIR /home/talkaa
WORKDIR /home/talkaa

COPY requirments.txt /${PROJECT_DIR}/requirements.txt
RUN apt-get -y update && \
    apt-get -y install apt-utils gcc curl && \
    pip install --no-cache-dir -r requirements.txt

COPY src /${PROJECT_DIR}/src
ENV MODEL_FILEPATH ${PROJECT_DIR}/src/model/file/iris_setosa.onnx

ADD run.sh /${PROJECT_DIR}/run.sh
RUN chmod +x /${PROJECT_DIR}/run.sh
CMD [ "./run.sh" ]
