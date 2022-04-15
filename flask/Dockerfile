FROM python:3.8.5

WORKDIR /src/app

RUN pip install flask
RUN pip install pandas
RUN pip install matplotlib
RUN pip install sklearn
RUN pip install xgboost
RUN pip install seaborn

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]