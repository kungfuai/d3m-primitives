FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9

COPY . /yonder-primitives

# We need to point DeepAR base library to more recent commit in git history
# Need to talk to Mitar about how to actually make this change - might have to first delete all TS primitives, 
# which would delete legacy TimeSeriesD3MWrappers and DeepAR base. 
RUN pip uninstall -y deepar 

# Might have to do first delete all legacy NK repos to delete installs...

RUN pip install -e ./yonder-primitives