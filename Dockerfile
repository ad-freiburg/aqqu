ARG TENSORFLOW=tensorflow/tensorflow:1.5.0-py3
FROM $TENSORFLOW
ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y python3-pip make libsnappy-dev zlib1g-dev libbz2-dev libgflags-dev librocksdb-dev

COPY requirements.txt /app/
WORKDIR /app/
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_lg

COPY . /app/
RUN python3 setup.py build_ext --inplace
VOLUME ["/app/models", "/app/data", "/app/input"]
ENTRYPOINT ["python3",  "-m"]
CMD ["query_translator.learner", "train", "WQSP_Ranker"]
