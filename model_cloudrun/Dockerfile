FROM gcc:9.3

RUN apt-get update
RUN apt-get install -y python3-dev \
	python3-pip \
	build-essential 

RUN python3 -m pip install numpy
RUN python3 -m pip install pillow
RUN python3 -m pip install flask
RUN python3 -m pip install flask-cors
RUN python3 -m pip install gunicorn
RUN python3 -m pip install tensorstore -vv
RUN python3 -m pip install tensorflow==2.0.0b1
RUN python3 -m pip install google-cloud-storage
RUN python3 -m pip install requests


WORKDIR /opt/app
COPY transferem.py ./ 
CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 transferem:app --timeout 900
