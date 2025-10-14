
.PHONY: all build clean create-textbook-stable-build create-textbook-dev-build

HNN_VERSION := 0.4.3
OS := $(shell uname -s)

# Function to create and configure a conda environment with library paths
# Usage: $(call create-and-configure-env,env-name,force-recreate)
# - env-name: name of the conda environment
# - force-recreate: if "true", deactivate and remove existing environment before creating
define create-and-configure-env
	@# If the second flag is true, then deactivate the env if it is present,
        @# then remove it if it exists.
	$(if $(filter true,$(2)),\
		-@if [ "$$CONDA_DEFAULT_ENV" = "$(1)" ]; then \
			echo "Deactivating environment '$(1)' ..."; \
			conda deactivate; \
		fi
		-@conda env list | grep -q "^$(1)\s" && \
		{ \
			echo "Attempting to remove '$(1)' environment"; \
			conda env remove -y -q --name $(1) && \
			echo "Environment '$(1)' successfully removed."; \
		} || true
	)
	@# Create the new conda environment from environment.yml
	conda env create --quiet --yes --file environment.yml --name $(1)
	@# Configure library paths for the environment
	@CONDA_ENV_PATH=$$(conda run -n $(1) python -c "import os; print(os.environ['CONDA_PREFIX'])"); \
	mkdir -p $$CONDA_ENV_PATH/etc/conda/activate.d ; \
	mkdir -p $$CONDA_ENV_PATH/etc/conda/deactivate.d ; \
	if [ "$(OS)" = "Darwin" ]; then \
		echo "export OLD_DYLD_FALLBACK_LIBRARY_PATH=\$$DYLD_FALLBACK_LIBRARY_PATH" >> "$$CONDA_ENV_PATH/etc/conda/activate.d/env_vars.sh"; \
		echo "export DYLD_FALLBACK_LIBRARY_PATH=\$$DYLD_FALLBACK_LIBRARY_PATH:$$CONDA_ENV_PATH/lib" >> "$$CONDA_ENV_PATH/etc/conda/activate.d/env_vars.sh"; \
		echo "export DYLD_FALLBACK_LIBRARY_PATH=\$$OLD_DYLD_FALLBACK_LIBRARY_PATH" >> "$$CONDA_ENV_PATH/etc/conda/deactivate.d/env_vars.sh"; \
		echo "unset OLD_DYLD_FALLBACK_LIBRARY_PATH" >> "$$CONDA_ENV_PATH/etc/conda/deactivate.d/env_vars.sh"; \
	elif [ "$(OS)" = "Linux" ]; then \
		echo "export OLD_LD_LIBRARY_PATH=\$$LD_LIBRARY_PATH" >> "$$CONDA_ENV_PATH/etc/conda/activate.d/env_vars.sh"; \
		echo "export LD_LIBRARY_PATH=\$$LD_LIBRARY_PATH:$$CONDA_ENV_PATH/lib" >> "$$CONDA_ENV_PATH/etc/conda/activate.d/env_vars.sh"; \
		echo "export LD_LIBRARY_PATH=\$$OLD_LD_LIBRARY_PATH" >> "$$CONDA_ENV_PATH/etc/conda/deactivate.d/env_vars.sh"; \
		echo "unset OLD_LD_LIBRARY_PATH" >> "$$CONDA_ENV_PATH/etc/conda/deactivate.d/env_vars.sh"; \
	fi;
endef

all: build

build:
	python build.py

force-execute-all-notebooks:
	python build.py --force-execute-all

execute-notebooks:
	python build.py --execute-notebooks

clean:
	rm -rf content/*.html
	rm -rf content/*/*.html

create-textbook-stable-build:
	$(call create-and-configure-env,textbook-stable-build,false)
	conda run -n textbook-stable-build pip install 'hnn_core[dev]==$(HNN_VERSION)'
	@echo "Conda environment 'textbook-stable-build' successfully created."
	@echo -e "\n\nActivate your environment with 'conda activate textbook-stable-build'"

create-textbook-dev-build:
	$(call create-and-configure-env,textbook-dev-build,true)

	@# Get the latest commit hash of hnn-core master branch
	LATEST_HASH=$$(git ls-remote https://github.com/jonescompneurolab/hnn-core.git master | cut -f1)

	@# Install hnn-core in developer mode, forcing reinstall without cache
	conda run -n textbook-dev-build pip install --upgrade --force-reinstall --no-cache-dir "hnn-core[dev] @ git+https://github.com/jonescompneurolab/hnn-core.git@master"

	@echo "Conda environment 'textbook-dev-build' successfully created."
	@echo -e "\n\nActivate your environment with 'conda activate textbook-dev-build'"

