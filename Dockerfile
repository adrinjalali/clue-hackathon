FROM tailordev/pandas

ADD . /

ENTRYPOINT ["/run.sh"]
