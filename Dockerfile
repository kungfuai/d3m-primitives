FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9

# remove old NK libaries
RUN rm -r /src/timeseriesd3mwrappers
RUN rm -r /src/simond3mwrapper
RUN rm -r /src/duked3mwrapper
RUN rm -r /src/datacleaningd3mwrapper
RUN rm -r /src/objectdetectiond3mwrapper
RUN rm -r /src/rffeaturesd3mwrapper
RUN rm -r /src/pcafeaturesd3mwrapper
COPY . /yonder-primitives

# We need to point DeepAR base library to more recent commit in git history
RUN pip uninstall -y deepar
#RUN pip install -e git+https://github.com/Yonder-OSS/D3M-Primitives@jg/image_primitives#egg=yonder-primitives --exists-action s
RUN pip install -e git+https://github.com/Yonder-OSS/D3M-Primitives@sn/migration#egg=yonder-primitives --exists-action s