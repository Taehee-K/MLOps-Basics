# FROM huggingface/transformers-pytorch-cpu:latest
FROM amazon/aws-lambda-python 

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models
RUN mkdir $MODEL_DIR

# add transformers cache to /models
ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY

RUN yum install git -y && yum -y install gcc-c++
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir 


COPY ./ ./
ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN export LC_ALL=C.UTF-8 && export LANG=C.UTF-8

# install requirements
# RUN pip install "dvc[gdrive]"
RUN pip install "dvc[s3]"   
# initialize dvc --no-scm to avoid git clone
RUN dvc init -f --no-scm 
# configuring remote server in dvc
RUN dvc remote add -d model-store s3://mlops-basics-model/trained_models/
# pull trained model
RUN dvc pull dvcfiles/trained_model.dvc

# # configuring remote server in dvc - google drive
# RUN dvc remote add -d storage gdrive://1o_fZ5FV6f515XNnM6oz5epogpXITLCc1
# RUN dvc remote modify storage gdrive_use_service_account true
# RUN dvc remote modify storage gdrive_service_account_json credentials.json

RUN ls 
RUN python lambda_handler.py 
RUN chmod -R 0755 $MODEL_DIR 
CMD [ "lambda_handler.lambda_handler"]

# # running the application
# EXPOSE 8000 
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]