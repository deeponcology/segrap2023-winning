FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN apt-get update -y
RUN apt-get install unzip
WORKDIR /home

RUN mkdir -p /home/SegRap23
RUN mkdir -p /home/nnUNet_trained_models/nnUNet/3d_fullres
RUN mkdir -p /home/TOTALSEG/nnUNet/3d_fullres
RUN mkdir -p /home/app
RUN mkdir -p /home/SegRap23/templates

RUN chmod -R 777 /home/SegRap23
RUN chmod -R 777 /home/nnUNet_trained_models/nnUNet/3d_fullres
RUN chmod -R 777 /home/TOTALSEG
RUN chmod -R 777 /home/app

# Totalseg stuff
ENV TOTALSEG_WEIGHTS_PATH=/home/TOTALSEG/

## Clone and Install totalsegmentor(and nnUNet v1)
RUN pip install TotalSegmentator

RUN adduser user
USER user



WORKDIR /home/SegRap23
RUN cd /home/SegRap23
## Copy SegRap Code
COPY templates/*.html /home/SegRap23/templates/
COPY requirements.txt /home/SegRap23/
COPY weight_dl.py /home/SegRap23/
COPY main.py /home/SegRap23/
COPY App.py /home/SegRap23/
COPY tools /home/SegRap23/tools/
## Install requirements
RUN pip install -r requirements.txt

## Download Trained Model and store it in /home/nnUNet_trained_models/nnUNet/3d_fullres
RUN cd /home/SegRap23
RUN python weight_dl.py
RUN unzip /home/app/Task606_SegRap1 -d /home/nnUNet_trained_models/nnUNet/3d_fullres
RUN unzip /home/app/Task269_Body_extrem_6mm_1200subj -d /home/TOTALSEG/nnUNet/3d_fullres
ENV RESULTS_FOLDER=/home/nnUNet_trained_models

RUN cd /home/SegRap23
# CMD ["python","main.py", "-i", "/input","-o","/output"]
ENV FLASK_APP=App.py
# RUN  /home/pipeline.sh 
# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
CMD ["python","main.py", "-i", "/input","-o","/output"]
