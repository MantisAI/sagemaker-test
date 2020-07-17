.DEFAULT_GOAL := all
PYTHON_VERSION := python3.7
ECR := 203149375586.dkr.ecr.eu-west-1.amazonaws.com

# Set the default location for the virtualenv to be stored

VIRTUALENV := build/virtualenv

# Create the virtualenv by installing the requirements and test requirements

$(VIRTUALENV)/.installed: requirements.txt
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON_VERSION) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	$(VIRTUALENV)/bin/pip3 install -r requirements_extra.txt
	touch $@

# Update the requirements to latest. This is required because typically we won't
# want to incldue test requirements in the requirements of the application, and
# because it makes life much easier when we want to keep our dependencies up to
# date.

.PHONY: update-requirements-txt
update-requirements-txt: VIRTUALENV := /tmp/update-requirements-virtualenv
update-requirements-txt:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON_VERSION) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r unpinned_requirements.txt
	echo "# Created by 'make update-requirements-txt'. DO NOT EDIT!" > requirements.txt
	$(VIRTUALENV)/bin/pip freeze | grep -v pkg-resources==0.0.0 >> requirements.txt

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed

.PHONY: docker-login
docker-login:
	aws ecr get-login-password --region=eu-west-1 | sudo  docker login \
		--username AWS --password-stdin ${ECR}

.env: .envrc
	sed -e "/export/!d" -e "s/export //g" $< > $@ 

all: virtualenv
