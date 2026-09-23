ARG ARENO_IMAGE
FROM ${ARENO_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends docker.io iptables \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install 'swebench==4.1.0' 'modelscope>=1.40.1' pyarrow
COPY examples/agentic/pi /opt/areno-pi
COPY examples/agentic/pi/dind/entrypoint.sh /usr/local/bin/areno-pi-dind
RUN chmod +x /usr/local/bin/areno-pi-dind
ENTRYPOINT ["/usr/local/bin/areno-pi-dind"]
CMD ["bash"]
