FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# this envs are experimental
ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY

# install requirements
# RUN pip install "dvc[gdrive]"
RUN pip install "dvc[s3]"   
RUN pip install -r requirements.txt

# # configuring remote server in dvc - google drive
# RUN dvc remote add -d storage gdrive://1o_fZ5FV6f515XNnM6oz5epogpXITLCc1
# RUN dvc remote modify storage gdrive_use_service_account true
# RUN dvc remote modify storage gdrive_service_account_json credentials.json

# initialize dvc
# RUN dvc init --no-scm
# configuring remote server in dvc
# RUN dvc remote add -d storage-s3 s3://mlops-basics-model/trained_models/

# RUN cat .dvc/config
# # pulling trained model
# RUN dvc pull dvcfiles/trained_model.dvc

RUN export LC_ALL=C.UTF-8 && export LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
EXPOSE 8000 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]