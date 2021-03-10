FROM gcc:9.3

RUN apt-get update
RUN apt-get install -y python3-dev \
	python3-pip \
	build-essential 

RUN python3 -m pip install pillow
RUN python3 -m pip install numpy
RUN python3 -m pip install flask
RUN python3 -m pip install flask-cors
RUN python3 -m pip install gunicorn
RUN python3 -m pip install tensorstore -vv

WORKDIR /opt/app
COPY transferem.py ./ 
CMD exec gunicorn --bind :$PORT --workers 9 --threads 2 transferem:app --timeout 900
