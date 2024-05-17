docker run --platform linux/amd64 -t --rm -p 8501:8501 \
    -v "./model/data/model:/models/windpower" \
    -e MODEL_NAME=windpower \
    tensorflow/serving &