FROM tailordev/pandas

ADD . /
RUN pip install -r requirements.txt
ENTRYPOINT ["/run.sh"]
