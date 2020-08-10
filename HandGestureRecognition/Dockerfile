FROM python:3.8.5

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY images /app/images/
COPY TrainedNeuralNetworks /app/TrainedNeuralNetworks/

COPY GetImages.py /app/
COPY NeuralNetworkTraining.py /app/
COPY HandGestureRecognition_Test.py /app/
COPY HandGestureRecognition.py /app/

CMD  bash
