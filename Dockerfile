FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9

# remove old NK libaries
RUN rm -r /src/simond3mwrapper
RUN rm -r /src/duked3mwrapper
RUN rm -r /src/datacleaningd3mwrapper
RUN rm -r /src/objectdetectiond3mwrapper
RUN rm -r /src/rffeaturesd3mwrapper
RUN rm -r /src/pcafeaturesd3mwrapper
RUN rm -r /src/gator
RUN rm -r /src/d3munsupervised
RUN rm -r /src/sent2vec-wrapper
RUN rm -r /src/goatd3mwrapper
RUN pip uninstall -y Sloth

#RUN pip install -e git+https://github.com/kungfuai/d3m-primitives#egg=primitives --exists-action s
