FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3
RUN apt-get update && apt-get install -y python3-pip make
# Mount point for integrating data volumes
RUN mkdir /data
RUN mkdir /app
WORKDIR /app/
COPY requirements-riseml-basics.txt /app/
RUN pip3 install -r requirements-riseml-basics.txt

# App specific rules follow
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

COPY . /app/
WORKDIR /app/
RUN python3 setup.py build_ext --inplace
CMD python -m query_translator.learner train WQ_Ranker
