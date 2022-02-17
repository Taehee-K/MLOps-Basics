FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

# install requirements
RUN pip install -r requirements.txt

# # configuring remote server in dvc
# RUN dvc remote add -d storage gdrive://1o_fZ5FV6f515XNnM6oz5epogpXITLCc1
# RUN dvc remote modify storage gdrive_use_service_account true
# RUN dvc remote modify storage gdrive_service_account_json credentials.json

# # pulling trained model
# RUN dvc pull dvcfiles/trained_model.dvc


RUN export LC_ALL=C.UTF-8 && export LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
EXPOSE 8000 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]