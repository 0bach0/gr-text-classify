FROM khanhtc3010/scilearn-c9:1.0
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app/
RUN pip install pyvi flask flask_restful
RUN tar -zxvf data.tar.gz 
RUN python create-model.py
EXPOSE 3000 80
CMD [ "python", "app.py" ]